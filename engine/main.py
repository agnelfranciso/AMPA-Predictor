import pandas as pd
import requests
import io
import datetime
import json
import math
import os
import sys
import argparse
import warnings
from scipy.stats import poisson, nbinom
import numpy as np
warnings.filterwarnings('ignore')

# ============================================================
#  AGNEL MATCH PREDICTION ENGINE (AMPE) v8.0
#  - Live scores from worldcup26.ir (free, no auth needed)
#  - WC2026 GROUP STAGE results injected into training data
#  - Deep football-analyst strengths/weaknesses (12+ dimensions)
#  - Versioned outputs + runs_index.js
#  - Dixon-Coles + H2H boost + Elo + form scoring + FIFA ranks
#  - v8.0: Negative Binomial dist, opponent-quality-adjusted stats,
#          momentum/streak detection, knockout pressure model,
#          venue/travel factor, fatigue/rest days, upgraded composite
# ============================================================

TARGET_DATE      = pd.to_datetime("2026-06-28")
HIST_URL         = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
LIVE_API_URL     = "https://worldcup26.ir/get/games"
FIFA_URL         = "https://raw.githubusercontent.com/tadhgfitzgerald/fifa_ranking/master/fifa_ranking.csv"
HALF_LIFE_DAYS   = 180
ELO_K_FACTOR     = 30
YEARS_BACK       = 6
FORM_GAMES       = 10
TOURNAMENT_BOOST = 1.2
# Scaling factor: inflate lambdas so predictions are more realistic (1.0 = base Poisson)
GOAL_INFLATE     = 1.25   # lifts avg goals from ~0.9 to ~1.1 per team per match
NBINOM_R         = 5.0    # Negative Binomial dispersion param (lower = fatter tails)
MOMENTUM_WIN_STREAK_THRESH  = 4   # consecutive wins for streak bonus
MOMENTUM_UNBEATEN_THRESH    = 8   # consecutive unbeaten for bonus
MOMENTUM_LOSE_STREAK_THRESH = 3   # consecutive losses for penalty

# ---- Confederation Mapping (for venue/travel factor) ----
CONFEDERATION = {
    "United States": "CONCACAF", "Canada": "CONCACAF", "Mexico": "CONCACAF",
    "Panama": "CONCACAF", "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "England": "UEFA", "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA",
    "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA",
    "Sweden": "UEFA", "Norway": "UEFA", "Scotland": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Morocco": "CAF", "Senegal": "CAF", "Ghana": "CAF", "Algeria": "CAF",
    "Ivory Coast": "CAF", "Egypt": "CAF", "South Africa": "CAF",
    "DR Congo": "CAF", "Cape Verde": "CAF",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Iran": "AFC",
    "Iraq": "AFC", "Uzbekistan": "AFC",
}

# ---- Ro32 Fixtures (display names) ----
fixtures_ro32 = [
    ("South Africa", "Canada"),
    ("Brazil",        "Japan"),
    ("Germany",       "Paraguay"),
    ("Netherlands",   "Morocco"),
    ("Côte d'Ivoire", "Norway"),
    ("France",        "Sweden"),
    ("Mexico",        "Ecuador"),
    ("England",       "Congo DR"),
    ("Belgium",       "Senegal"),
    ("USA",           "Bosnia and Herzegovina"),
    ("Spain",         "Austria"),
    ("Portugal",      "Croatia"),
    ("Switzerland",   "Algeria"),
    ("Australia",     "Egypt"),
    ("Argentina",     "Cabo Verde"),
    ("Colombia",      "Ghana"),
]

TEAM_NAME_MAP = {
    "USA":            "United States",
    "Congo DR":       "DR Congo",
    "Cabo Verde":     "Cape Verde",
    "Côte d'Ivoire":  "Ivory Coast",
}
REVERSE_MAP = {v: k for k, v in TEAM_NAME_MAP.items()}

# Mapping display names -> possible API name variants
LIVE_API_NAME_MAP = {
    "South Africa":          ["South Africa"],
    "Canada":                ["Canada"],
    "Brazil":                ["Brazil"],
    "Japan":                 ["Japan"],
    "Germany":               ["Germany"],
    "Paraguay":              ["Paraguay"],
    "Netherlands":           ["Netherlands","Holland"],
    "Morocco":               ["Morocco"],
    "Côte d'Ivoire":         ["Ivory Coast","Côte d'Ivoire","Cote d'Ivoire","Cote dIvoire","CI"],
    "Norway":                ["Norway"],
    "France":                ["France"],
    "Sweden":                ["Sweden"],
    "Mexico":                ["Mexico"],
    "Ecuador":               ["Ecuador"],
    "England":               ["England"],
    "Congo DR":              ["DR Congo","Congo DR","Congo-Kinshasa","Democratic Republic of Congo","Democratic Republic of the Congo"],
    "Belgium":               ["Belgium"],
    "Senegal":               ["Senegal"],
    "USA":                   ["USA","United States","US"],
    "Bosnia and Herzegovina":["Bosnia","Bosnia and Herzegovina","Bosnia & Herzegovina"],
    "Spain":                 ["Spain"],
    "Austria":               ["Austria"],
    "Portugal":              ["Portugal"],
    "Croatia":               ["Croatia"],
    "Switzerland":           ["Switzerland"],
    "Algeria":               ["Algeria"],
    "Australia":             ["Australia"],
    "Egypt":                 ["Egypt"],
    "Argentina":             ["Argentina"],
    "Cabo Verde":            ["Cape Verde","Cabo Verde"],
    "Colombia":              ["Colombia"],
    "Ghana":                 ["Ghana"],
}

def ds(name):
    return TEAM_NAME_MAP.get(name, name)

def dn(ds_name):
    return REVERSE_MAP.get(ds_name, ds_name)

def map_team_name(name):
    if not name or name == '0':
        return None
    for display, variants in LIVE_API_NAME_MAP.items():
        if any(v.lower() in name.lower() or name.lower() in v.lower() for v in variants):
            return display
    return name

# ============================================================
#  LIVE SCHEDULE & SCORES from worldcup26.ir
# ============================================================
def fetch_wc2026_games():
    """Fetch the full list of matches from the API or fallback to local api.json."""
    print("Fetching WC2026 match schedule from worldcup26.ir...")
    try:
        resp = requests.get(LIVE_API_URL, timeout=12)
        if resp.status_code == 200:
            raw = resp.json()
            games = raw if isinstance(raw, list) else raw.get('games', raw.get('data', []))
            return games
    except Exception as e:
        print(f"  Error fetching match schedule: {e}")
    
    print("  Falling back to local api.json...")
    try:
        with open('api.json', 'r', encoding='utf-8') as f:
            raw = json.load(f)
            games = raw if isinstance(raw, list) else raw.get('games', raw.get('data', []))
            return games
    except Exception as e:
        print(f"  Error reading local api.json: {e}")
    return []

def fetch_live_scores(games):
    """Fetch live/finished match results from pre-loaded games list.
    Returns a dict keyed by frozenset of team display names -> {score, status, home, away}
    """
    print("Parsing live WC2026 scores...")

    live_scores = {}
    count = 0
    for game in games:
        # API field names can vary - try multiple
        home = (game.get('homeTeam') or game.get('home_team') or game.get('home') or {})
        away = (game.get('awayTeam') or game.get('away_team') or game.get('away') or {})
        
        if isinstance(home, str): home_name = home
        else: home_name = home.get('name') or home.get('en') or home.get('nameEn') or ''
        if not home_name: home_name = game.get('home_team_name_en') or ''
        
        if isinstance(away, str): away_name = away
        else: away_name = away.get('name') or away.get('en') or away.get('nameEn') or ''
        if not away_name: away_name = game.get('away_team_name_en') or ''

        status = (game.get('status') or game.get('matchStatus') or game.get('time_elapsed') or str(game.get('finished'))).upper()
        if status == 'TRUE': status = 'FINISHED'
        
        hs = game.get('homeScore') or game.get('home_score') or game.get('homeGoals') or game.get('score', {}).get('home', None)
        as_ = game.get('awayScore') or game.get('away_score') or game.get('awayGoals') or game.get('score', {}).get('away', None)

        if not home_name or not away_name:
            continue

        # Map API names to our display names
        mapped_home = None
        mapped_away = None
        for display, variants in LIVE_API_NAME_MAP.items():
            if any(v.lower() in home_name.lower() or home_name.lower() in v.lower() for v in variants):
                mapped_home = display
            if any(v.lower() in away_name.lower() or away_name.lower() in v.lower() for v in variants):
                mapped_away = display

        if mapped_home and mapped_away:
            # Skip group stage matches so they don't accidentally override a knockout match between the same teams!
            if game.get('type') == 'group':
                continue
                
            key = frozenset([mapped_home, mapped_away])
            is_finished = any(w in status for w in ['FINISH','COMPLET','ENDED','FULL','FT'])
            score_available = (hs not in [None, 'null', 'NULL', ''] and as_ not in [None, 'null', 'NULL', ''])
            
            try:
                hs_int = int(hs) if score_available else None
                as_int = int(as_) if score_available else None
            except (ValueError, TypeError):
                hs_int = None
                as_int = None
                score_available = False

            hp = game.get('home_penalty_score')
            ap = game.get('away_penalty_score')
            hp_int = None
            ap_int = None
            if hp not in [None, 'null', 'NULL', '']:
                try: hp_int = int(hp)
                except ValueError: pass
            if ap not in [None, 'null', 'NULL', '']:
                try: ap_int = int(ap)
                except ValueError: pass

            # Parse kickoff date
            kickoff_str = game.get('local_date', '')
            try:
                dt = datetime.datetime.strptime(kickoff_str, "%m/%d/%Y %H:%M")
                kickoff_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                kickoff_iso = kickoff_str

            live_scores[key] = {
                'home': mapped_home,
                'away': mapped_away,
                'home_score': hs_int,
                'away_score': as_int,
                'home_pen': hp_int,
                'away_pen': ap_int,
                'status': status,
                'finished': is_finished and score_available,
                'kickoff': kickoff_iso,
            }
            count += 1

    print(f"  Found {count} mapped matches from live API, {sum(1 for v in live_scores.values() if v['finished'])} finished.")
    return live_scores

# ============================================================
#  HISTORICAL DATA
# ============================================================
# ============================================================
#  WC2026 HISTORY INJECTOR
#  Maps worldcup26.ir team names -> martj42-compatible names
# ============================================================
WC26_TO_MARTJ42 = {
    "Mexico":                          "Mexico",
    "South Africa":                    "South Africa",
    "South Korea":                     "South Korea",
    "Czech Republic":                  "Czech Republic",
    "Canada":                          "Canada",
    "Bosnia and Herzegovina":          "Bosnia and Herzegovina",
    "Qatar":                           "Qatar",
    "Switzerland":                     "Switzerland",
    "Haiti":                           "Haiti",
    "Scotland":                        "Scotland",
    "United States":                   "United States",
    "Paraguay":                        "Paraguay",
    "Turkey":                          "Turkey",
    "Australia":                       "Australia",
    "Brazil":                          "Brazil",
    "Morocco":                         "Morocco",
    "Ivory Coast":                     "Ivory Coast",
    "Ecuador":                         "Ecuador",
    "Germany":                         "Germany",
    "Curaçao":                         "Curacao",
    "Netherlands":                     "Netherlands",
    "Japan":                           "Japan",
    "Sweden":                          "Sweden",
    "Tunisia":                         "Tunisia",
    "Belgium":                         "Belgium",
    "Iran":                            "Iran",
    "Egypt":                           "Egypt",
    "New Zealand":                     "New Zealand",
    "Spain":                           "Spain",
    "Cape Verde":                      "Cape Verde",
    "Saudi Arabia":                    "Saudi Arabia",
    "Uruguay":                         "Uruguay",
    "France":                          "France",
    "Senegal":                         "Senegal",
    "Iraq":                            "Iraq",
    "Norway":                          "Norway",
    "England":                         "England",
    "Croatia":                         "Croatia",
    "Panama":                          "Panama",
    "Ghana":                           "Ghana",
    "Portugal":                        "Portugal",
    "Democratic Republic of the Congo":"DR Congo",
    "Uzbekistan":                      "Uzbekistan",
    "Colombia":                        "Colombia",
    "Algeria":                         "Algeria",
    "Austria":                         "Austria",
    "Jordan":                          "Jordan",
    "Argentina":                       "Argentina",
}

def fetch_wc2026_history(games):
    """Filter all FINISHED WC2026 group stage matches from games
    and return a DataFrame compatible with the martj42 dataset."""
    print("Processing WC2026 completed matches for training data...")
    rows = []
    for g in games:
        if str(g.get('finished', '')).upper() != 'TRUE':
            continue
        home_en = g.get('home_team_name_en', '')
        away_en = g.get('away_team_name_en', '')
        if not home_en or not away_en:
            continue
        home_m = WC26_TO_MARTJ42.get(home_en)
        away_m = WC26_TO_MARTJ42.get(away_en)
        if not home_m or not away_m:
            continue
        try:
            hs = int(g.get('home_score', 0))
            as_ = int(g.get('away_score', 0))
        except:
            continue
        # Parse date from local_date like "06/13/2026 21:00"
        raw_date = g.get('local_date', '')
        try:
            dt = pd.to_datetime(raw_date, format='%m/%d/%Y %H:%M')
        except:
            continue
        gtype = g.get('type', 'group')
        tourn = 'FIFA World Cup' if 'group' in gtype else 'FIFA World Cup KO'
        rows.append({'date': dt, 'home_team': home_m, 'away_team': away_m,
                     'home_score': hs, 'away_score': as_,
                     'tournament': tourn, 'city': '', 'country': '', 'neutral': True})

    if not rows:
        print("  No finished WC2026 matches found to inject")
        return None

    df_wc = pd.DataFrame(rows)
    print(f"  Injecting {len(df_wc)} finished WC2026 group stage matches into training data")
    return df_wc


def load_data(games=None):
    print("Downloading historical match data (martj42 dataset)...")
    resp = requests.get(HIST_URL, timeout=30)
    df = pd.read_csv(io.StringIO(resp.text))
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    hist_last = df['date'].max()
    print(f"  Loaded {len(df):,} matches ({df['date'].min().year}–{hist_last.year})")
    print(f"  Dataset last updated: {hist_last.strftime('%Y-%m-%d')}")

    # Inject WC2026 live match results (fills gap after martj42 cutoff)
    if games is None:
        games = fetch_wc2026_games()
    df_wc = fetch_wc2026_history(games)
    if df_wc is not None:
        # Remove any WC2026 matches already in martj42 to avoid duplicates
        wc_cutoff = df_wc['date'].min()
        df_existing = df[df['date'] < wc_cutoff]
        df = pd.concat([df_existing, df_wc], ignore_index=True)
        # Also add WC2026 rows that are after martj42's cutoff
        df_after = df_wc[df_wc['date'] > hist_last]
        if len(df_after):
            df = pd.concat([df, df_after], ignore_index=True)
            df = df.drop_duplicates(subset=['date','home_team','away_team'], keep='last')
        print(f"  Combined dataset: {len(df):,} matches, up to {df['date'].max().strftime('%Y-%m-%d')}")
    return df

# ============================================================
#  WEIGHTING
# ============================================================
def match_weight(row, target_date):
    days_ago = max((target_date - row['date']).days, 0)
    w = math.exp(-math.log(2) * days_ago / HALF_LIFE_DAYS)
    kwds = ['FIFA World Cup','UEFA Euro','Copa América','African Cup','AFC Asian Cup','CONCACAF']
    return w * (TOURNAMENT_BOOST if any(k in str(row.get('tournament','')) for k in kwds) else 1.0)

# ============================================================
#  FIFA RANKINGS
# ============================================================
def fetch_fifa_rankings():
    print("Fetching official FIFA rankings...")
    try:
        resp = requests.get(FIFA_URL, timeout=12)
        if resp.status_code == 200:
            df_fifa = pd.read_csv(io.StringIO(resp.text))
            # Sort by rank_date descending and take the most recent for each country
            df_fifa = df_fifa.sort_values('rank_date', ascending=False).drop_duplicates(subset=['country_full'])
            rankings = {}
            for _, row in df_fifa.iterrows():
                # Store points, rank, etc.
                rankings[row['country_full']] = {
                    'points': float(row['total_points']),
                    'rank': int(row['rank'])
                }
            print(f"  Loaded FIFA rankings for {len(rankings)} teams.")
            return rankings
        else:
            print(f"  Failed to fetch FIFA rankings (HTTP {resp.status_code})")
    except Exception as e:
        print(f"  Error fetching FIFA rankings: {e}")
    return {}

# ============================================================
#  ELO
# ============================================================
def compute_elo(df, target_date):
    print("Computing Elo ratings...")
    elo = {}
    for _, row in df[df['date'] <= target_date].sort_values('date').iterrows():
        ht, at = row['home_team'], row['away_team']
        hs, as_ = row['home_score'], row['away_score']
        rh, ra = elo.get(ht, 1500.0), elo.get(at, 1500.0)
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        ea = 1 - eh
        sh, sa = (1.0,0.0) if hs>as_ else ((0.0,1.0) if hs<as_ else (0.5,0.5))
        gd = abs(hs-as_)
        gm = 1.0 if gd<=1 else (1.5 if gd==2 else (1.75 if gd==3 else 1.75+0.25*(gd-3)))
        
        tourn = str(row.get('tournament', '')).lower()
        if 'friendly' in tourn:
            k_factor = 20
        elif 'qualification' in tourn or 'qualifier' in tourn:
            k_factor = 40
        elif 'world cup' in tourn and 'qualification' not in tourn:
            k_factor = 60
        elif any(t in tourn for t in ['euro', 'copa am', 'africa cup', 'asian cup', 'gold cup']):
            k_factor = 50
        else:
            k_factor = ELO_K_FACTOR # Default 30 for minor tournaments
            
        elo[ht] = rh + k_factor*gm*(sh-eh)
        elo[at] = ra + k_factor*gm*(sa-ea)
    print(f"  Elo computed for {len(elo):,} teams")
    return elo

# ============================================================
#  ATTACK / DEFENSE STRENGTHS (v8.0: opponent-quality adjusted)
# ============================================================
def compute_strengths(df, target_date):
    print("Computing attack/defense strengths (time-decay weighted, opponent-quality adjusted)...")
    cutoff = target_date - pd.Timedelta(days=365*YEARS_BACK)
    dfw = df[(df['date']>=cutoff)&(df['date']<target_date)].copy()
    dfw['w'] = dfw.apply(lambda r: match_weight(r, target_date), axis=1)
    tw = dfw['w'].sum()
    global_avg = ((dfw['home_score']*dfw['w']).sum() + (dfw['away_score']*dfw['w']).sum()) / (2*tw)
    teams = set(dfw['home_team'])|set(dfw['away_team'])

    # First pass: raw stats (needed for opponent quality lookup)
    raw_stats = {}
    for team in teams:
        hm = dfw[dfw['home_team']==team]
        am = dfw[dfw['away_team']==team]
        tw_ = hm['w'].sum()+am['w'].sum()
        if tw_ < 0.01:
            raw_stats[team] = {'attack':1.0,'defense':1.0,'avg_gf':global_avg,'avg_ga':global_avg,'n':0}
            continue
        gf = (hm['home_score']*hm['w']).sum()+(am['away_score']*am['w']).sum()
        ga = (hm['away_score']*hm['w']).sum()+(am['home_score']*am['w']).sum()
        agf, aga = gf/tw_, ga/tw_
        raw_stats[team] = {
            'attack':  round(agf/global_avg, 4) if global_avg>0 else 1.0,
            'defense': round(aga/global_avg, 4) if global_avg>0 else 1.0,
            'avg_gf':  round(agf, 3),
            'avg_ga':  round(aga, 3),
            'n':       len(hm)+len(am),
        }

    # Second pass: adjust for opponent quality
    # Scoring 3 against a team with defense=0.5 is worth more than scoring 3 against defense=1.5
    stats = {}
    for team in teams:
        hm = dfw[dfw['home_team']==team]
        am = dfw[dfw['away_team']==team]
        tw_ = hm['w'].sum()+am['w'].sum()
        if tw_ < 0.01:
            stats[team] = raw_stats[team]
            continue

        adj_gf = 0.0
        adj_ga = 0.0
        for _, row in hm.iterrows():
            opp = row['away_team']
            opp_def = raw_stats.get(opp, {}).get('defense', 1.0)
            opp_att = raw_stats.get(opp, {}).get('attack', 1.0)
            # Goals scored against a strong defense (low defense index) are worth more
            adj_gf += row['home_score'] * row['w'] * max(0.5, min(2.0, 1.0 / max(opp_def, 0.3)))
            # Goals conceded from a strong attack are less punishing
            adj_ga += row['away_score'] * row['w'] * max(0.5, min(2.0, 1.0 / max(opp_att, 0.3)))
        for _, row in am.iterrows():
            opp = row['home_team']
            opp_def = raw_stats.get(opp, {}).get('defense', 1.0)
            opp_att = raw_stats.get(opp, {}).get('attack', 1.0)
            adj_gf += row['away_score'] * row['w'] * max(0.5, min(2.0, 1.0 / max(opp_def, 0.3)))
            adj_ga += row['home_score'] * row['w'] * max(0.5, min(2.0, 1.0 / max(opp_att, 0.3)))

        adj_agf = adj_gf / tw_
        adj_aga = adj_ga / tw_

        # Blend: 60% opponent-adjusted, 40% raw (to avoid over-correction)
        raw = raw_stats[team]
        blend_gf = 0.6 * adj_agf + 0.4 * raw['avg_gf']
        blend_ga = 0.6 * adj_aga + 0.4 * raw['avg_ga']

        stats[team] = {
            'attack':  round(blend_gf/global_avg, 4) if global_avg>0 else 1.0,
            'defense': round(blend_ga/global_avg, 4) if global_avg>0 else 1.0,
            'avg_gf':  round(blend_gf, 3),
            'avg_ga':  round(blend_ga, 3),
            'n':       raw['n'],
        }
    print(f"  Strengths computed for {len(stats):,} teams | global avg goals/team: {global_avg:.3f}")
    return stats, global_avg

# ============================================================
#  FORM
# ============================================================
def compute_form(df, ds_name, target_date, n=FORM_GAMES):
    ms = df[((df['home_team']==ds_name)|(df['away_team']==ds_name))&(df['date']<target_date)].sort_values('date',ascending=False).head(n)
    results, pts = [], []
    for _, m in ms.iterrows():
        ih = m['home_team']==ds_name
        gf = m['home_score'] if ih else m['away_score']
        gc = m['away_score'] if ih else m['home_score']
        opp = m['away_team'] if ih else m['home_team']
        r = 'W' if gf>gc else ('D' if gf==gc else 'L')
        pts.append(3 if r=='W' else (1 if r=='D' else 0))
        results.append({'date':m['date'].strftime('%Y-%m-%d'),'opponent':dn(opp),'gf':int(gf),'gc':int(gc),'result':r,'tournament':str(m.get('tournament',''))})
    if not pts: return 50.0, [], "No recent data"
    ws = [math.exp(-0.15*i) for i in range(len(pts))]
    score = round(sum((p/3)*w for p,w in zip(pts,ws))/sum(ws)*100, 1)
    lg = results[0]
    last = {'W':'Won','D':'Drew','L':'Lost'}[lg['result']] + f" {lg['gf']}-{lg['gc']} vs {lg['opponent']} ({lg['date']})"
    return score, results, last

# ============================================================
#  MOMENTUM & STREAK DETECTION (v8.0)
# ============================================================
def compute_momentum(df, ds_name, target_date, n=15):
    """Detect winning/losing/unbeaten streaks and scoring streaks.
    Returns a dict with streak info and a momentum multiplier."""
    ms = df[((df['home_team']==ds_name)|(df['away_team']==ds_name))&(df['date']<target_date)].sort_values('date',ascending=False).head(n)
    if len(ms) == 0:
        return {'win_streak': 0, 'unbeaten_streak': 0, 'lose_streak': 0, 'scoring_streak': 0, 'multiplier': 1.0}

    win_streak = 0
    unbeaten_streak = 0
    lose_streak = 0
    scoring_streak = 0
    for _, m in ms.iterrows():
        ih = m['home_team'] == ds_name
        gf = m['home_score'] if ih else m['away_score']
        gc = m['away_score'] if ih else m['home_score']
        # Win streak
        if gf > gc:
            if lose_streak == 0: win_streak += 1
            else: break
        elif gf == gc:
            if lose_streak == 0 and win_streak > 0: pass  # unbeaten continues
            else: break
        else:
            break

    # Recalculate unbeaten and lose streaks separately
    for _, m in ms.iterrows():
        ih = m['home_team'] == ds_name
        gf = m['home_score'] if ih else m['away_score']
        gc = m['away_score'] if ih else m['home_score']
        if gf >= gc: unbeaten_streak += 1
        else: break

    for _, m in ms.iterrows():
        ih = m['home_team'] == ds_name
        gf = m['home_score'] if ih else m['away_score']
        gc = m['away_score'] if ih else m['home_score']
        if gf < gc: lose_streak += 1
        else: break

    # Scoring streak (scored 2+ goals)
    for _, m in ms.iterrows():
        ih = m['home_team'] == ds_name
        gf = m['home_score'] if ih else m['away_score']
        if gf >= 2: scoring_streak += 1
        else: break

    # Calculate multiplier
    mult = 1.0
    if win_streak >= MOMENTUM_WIN_STREAK_THRESH:
        mult += 0.02 * min(win_streak - MOMENTUM_WIN_STREAK_THRESH + 1, 4)  # +2% to +8%
    if unbeaten_streak >= MOMENTUM_UNBEATEN_THRESH:
        mult += 0.01 * min(unbeaten_streak - MOMENTUM_UNBEATEN_THRESH + 1, 4)  # +1% to +4%
    if lose_streak >= MOMENTUM_LOSE_STREAK_THRESH:
        mult -= 0.03 * min(lose_streak - MOMENTUM_LOSE_STREAK_THRESH + 1, 3)  # -3% to -9%
    if scoring_streak >= 5:
        mult += 0.015 * min(scoring_streak - 4, 3)  # +1.5% to +4.5% attack boost

    return {
        'win_streak': win_streak,
        'unbeaten_streak': unbeaten_streak,
        'lose_streak': lose_streak,
        'scoring_streak': scoring_streak,
        'multiplier': round(mult, 4)
    }

# ============================================================
#  KNOCKOUT PRESSURE MODEL (v8.0)
# ============================================================
def compute_knockout_factor(df, ds_name, target_date):
    """Compute a knockout stage pressure factor based on historical
    performance in knockout/elimination rounds of major tournaments."""
    ko_keywords = ['knockout', 'final', 'quarter', 'semi', 'round of']
    tourn_keywords = ['FIFA World Cup', 'UEFA Euro', 'Copa América', 'Africa Cup', 'Asian Cup']

    tdf = df[
        (df['tournament'].apply(lambda t: any(k in str(t) for k in tourn_keywords))) &
        ((df['home_team']==ds_name)|(df['away_team']==ds_name)) &
        (df['date']<target_date) &
        (df['date']>=target_date-pd.Timedelta(days=365*12))
    ]

    # Count wins/losses in elimination-style matches (proxy: not group stage)
    # We use a simple heuristic: matches in major tournaments that are NOT in a group stage
    # Major tournament matches where teams play each other only once tend to be knockouts
    w = l = d = 0
    penalty_wins = 0
    penalty_losses = 0
    for _, m in tdf.iterrows():
        ih = m['home_team'] == ds_name
        gf = m['home_score'] if ih else m['away_score']
        gc = m['away_score'] if ih else m['home_score']
        if gf > gc: w += 1
        elif gf < gc: l += 1
        else:
            d += 1
            # Draws in knockout = likely penalties
            penalty_wins += 0.5  # assume 50/50 if we don't know

    total = max(w + l + d, 1)
    win_rate = (w + 0.3 * d) / total  # draws less valuable in knockouts

    # Knockout factor: ranges from 0.95 (poor KO record) to 1.08 (excellent KO record)
    factor = 0.95 + 0.13 * min(1.0, win_rate / 0.6)

    return {
        'ko_wins': w, 'ko_losses': l, 'ko_draws': d,
        'ko_win_rate': round(win_rate * 100, 1),
        'factor': round(factor, 4)
    }

# ============================================================
#  VENUE & TRAVEL FACTOR (v8.0)
# ============================================================
def compute_venue_factor(ds_name):
    """Calculate venue/travel advantage. Tournament is in USA/Canada/Mexico."""
    conf = CONFEDERATION.get(ds_name, 'OTHER')
    HOSTS = {'United States', 'Canada', 'Mexico'}
    if ds_name in HOSTS:
        return 1.12  # Strong home advantage
    elif conf == 'CONCACAF':
        return 1.05  # Familiar conditions, nearby travel
    elif conf == 'CONMEBOL':
        return 1.03  # Americas, moderate travel, similar time zones
    elif conf == 'UEFA':
        return 1.00  # Neutral — used to big tournaments, moderate travel
    elif conf == 'CAF':
        return 0.98  # Significant travel, less familiar conditions
    elif conf == 'AFC':
        return 0.96  # Long-haul, jet lag, very different conditions
    return 0.98  # Default slight disadvantage

# ============================================================
#  GOAL STATS
# ============================================================
def compute_goal_stats(df, ds_name, target_date, n=20):
    ms = df[((df['home_team']==ds_name)|(df['away_team']==ds_name))&(df['date']<target_date)].sort_values('date',ascending=False).head(n)
    cs=btts=gf_t=ga_t=0
    for _,m in ms.iterrows():
        ih = m['home_team']==ds_name
        gf = m['home_score'] if ih else m['away_score']
        ga = m['away_score'] if ih else m['home_score']
        gf_t+=gf; ga_t+=ga
        if ga==0: cs+=1
        if gf>0 and ga>0: btts+=1
    n_ = max(len(ms),1)
    return {'clean_sheets':cs,'clean_sheet_pct':round(cs/n_*100,1),'btts_pct':round(btts/n_*100,1),'avg_gf':round(gf_t/n_,2),'avg_ga':round(ga_t/n_,2),'n':n_}

# ============================================================
#  TOURNAMENT RECORD
# ============================================================
def compute_tourn_record(df, ds_name, target_date):
    kwds = ['FIFA World Cup','UEFA Euro','Copa América','Africa Cup','Asian Cup','CONCACAF Gold Cup']
    tdf = df[df['tournament'].apply(lambda t:any(k in str(t) for k in kwds))&
             ((df['home_team']==ds_name)|(df['away_team']==ds_name))&
             (df['date']<target_date)&(df['date']>=target_date-pd.Timedelta(days=365*10))]
    w=d=l=gf=ga=0
    for _,m in tdf.iterrows():
        ih=m['home_team']==ds_name
        mgs=m['home_score'] if ih else m['away_score']; mga=m['away_score'] if ih else m['home_score']
        gf+=mgs; ga+=mga
        if mgs>mga: w+=1
        elif mgs==mga: d+=1
        else: l+=1
    return {'wins':w,'draws':d,'losses':l,'gf':gf,'ga':ga,'matches':w+d+l}

# ============================================================
#  H2H
# ============================================================
def compute_h2h(df, ds1, ds2, target_date=None):
    h2h = df[((df['home_team']==ds1)&(df['away_team']==ds2))|((df['home_team']==ds2)&(df['away_team']==ds1))].sort_values('date',ascending=False)
    t1w=t2w=draws=t1gf=t1ga=0
    weighted_t1gf=0.0
    weighted_t1ga=0.0
    weighted_total=0.0
    meetings=[]
    for _,m in h2h.iterrows():
        ih=m['home_team']==ds1
        gf=m['home_score'] if ih else m['away_score']; gc=m['away_score'] if ih else m['home_score']
        t1gf+=gf; t1ga+=gc
        
        # Calculate time-decay weight for this encounter (365 days half-life)
        if target_date is not None:
            days_ago = max((target_date - m['date']).days, 0)
            w = math.exp(-math.log(2) * days_ago / 365.0)
        else:
            w = 1.0
            
        weighted_t1gf += gf * w
        weighted_t1ga += gc * w
        weighted_total += w
        
        if gf>gc: t1w+=1; r='W'
        elif gf<gc: t2w+=1; r='L'
        else: draws+=1; r='D'
        meetings.append({'date':m['date'].strftime('%Y-%m-%d'),'score':f"{gf}-{gc}",'result':r,'tourn':str(m.get('tournament',''))})
    return {
        't1_wins':t1w,'t2_wins':t2w,'draws':draws,'total':t1w+t2w+draws,
        't1_gf':t1gf,'t1_ga':t1ga,
        'weighted_t1_gf': weighted_t1gf,
        'weighted_t1_ga': weighted_t1ga,
        'weighted_total': weighted_total,
        'meetings':meetings[:10]
    }

# ============================================================
#  STRENGTHS / WEAKNESSES  (deep football analyst model)
# ============================================================
def swlabels(attack, defense, form, gs, tourn, elo, form_results=None):
    if form_results is None: form_results = []
    S, W = [], []
    gf   = gs.get('avg_gf', 0)
    ga   = gs.get('avg_ga', 0)
    cs   = gs.get('clean_sheet_pct', 0)
    btts = gs.get('btts_pct', 0)
    n    = max(len(form_results), 1)
    wins   = sum(1 for r in form_results if r.get('result')=='W')
    draws  = sum(1 for r in form_results if r.get('result')=='D')
    losses = sum(1 for r in form_results if r.get('result')=='L')
    win_r  = wins / n * 100
    loss_r = losses / n * 100

    # --- ATTACK ---
    if   attack >= 1.8:  S.append(f"Devastating, world-class attack \u2014 avg {gf:.1f} goals/game, relentlessly creates chances")
    elif attack >= 1.45: S.append(f"Lethal and prolific attack \u2014 scores freely at {gf:.1f} g/game, well above global avg")
    elif attack >= 1.15: S.append(f"Sharp, effective attack \u2014 consistently creates and converts ({gf:.1f} g/game)")
    elif attack <= 0.55: W.append(f"Severely blunt attack \u2014 only {gf:.1f} g/game, one of the lowest-scoring teams")
    elif attack <= 0.80: W.append(f"Struggles to score \u2014 {gf:.1f} g/game, lacks cutting edge in the final third")
    elif attack <= 0.92: W.append(f"Below-average firepower \u2014 often fails to convert pressure into goals ({gf:.1f} g/game)")

    # --- DEFENSE ---
    if   defense <= 0.50: S.append(f"Fortress-level defense \u2014 concedes just {ga:.1f} g/game, elite defensive discipline")
    elif defense <= 0.72: S.append(f"Elite backline \u2014 one of the strongest defensive units at this tournament ({ga:.1f} g/game)")
    elif defense <= 0.88: S.append(f"Compact and well-organized defense \u2014 hard to break down ({ga:.1f} conceded/game)")
    elif defense >= 1.55: W.append(f"Very porous defense \u2014 concedes {ga:.1f} g/game, routinely exploited by good teams")
    elif defense >= 1.25: W.append(f"Defensive fragility \u2014 {ga:.1f} g/game conceded, backline regularly under pressure")
    elif defense >= 1.08: W.append(f"Defensively uncertain \u2014 can be opened up by quality attacking play ({ga:.1f} g/game)")

    # --- TACTICAL STYLE inferred from gf/ga pattern ---
    if gf >= 2.1 and ga >= 1.6:
        S.append("High-octane, free-flowing style \u2014 games tend to be open and entertaining")
        W.append("Open play style leaves gaps \u2014 vulnerable to lethal counter-attacking teams")
    elif gf >= 1.9 and ga <= 0.85:
        S.append("Complete team \u2014 dominates both ends; scores freely and keeps opponents at bay")
    elif gf <= 0.85 and ga <= 0.85 and cs >= 40:
        S.append("Defensive masterclass \u2014 disciplined low-block, forces opponents to work hard for everything")
        W.append("Risk-averse and pragmatic \u2014 can struggle to break down equally defensive setups")
    elif gf <= 0.85 and ga >= 1.35:
        W.append("Lacks a clear identity \u2014 neither scores reliably nor defends convincingly")

    # --- CLEAN SHEETS ---
    if   cs >= 55: S.append(f"Exceptional clean sheet specialist \u2014 {cs}% of recent games kept opponents scoreless")
    elif cs >= 38: S.append(f"Solid defensive record \u2014 {cs}% clean sheet rate shows strong organization")

    # --- BTTS PATTERN ---
    if   btts >= 65: W.append(f"Both teams score in {btts}% of their games \u2014 defensive structure is permeable")
    elif btts <= 22 and cs >= 38: S.append("Dominant defensively \u2014 opposition rarely scores; controls games well")

    # --- FORM ---
    if   form >= 88: S.append(f"Outstanding current form \u2014 near-perfect results, full of confidence and momentum")
    elif form >= 72: S.append(f"Excellent recent form \u2014 {wins}W-{draws}D-{losses}L in last {n} games, riding high")
    elif form >= 58: S.append(f"Positive recent run \u2014 more wins than defeats, growing confidence")
    elif form <= 25: W.append(f"Alarming form \u2014 only {wins} wins from {n} recent games; severe confidence crisis")
    elif form <= 40: W.append(f"Poor recent run \u2014 {losses} defeats from {n} games, lacking energy and conviction")
    elif form <= 52: W.append(f"Inconsistent form \u2014 too many draws ({draws} from {n}), unable to turn pressure into wins")

    # --- CONSISTENCY ---
    if win_r >= 70 and n >= 8:
        S.append(f"Highly consistent winners \u2014 {wins} wins from {n} recent games shows reliable excellence")
    if loss_r >= 50 and n >= 6:
        W.append(f"Alarming inconsistency \u2014 {losses} losses from {n} games; cannot be relied upon")
    if draws >= 5 and n >= 8:
        W.append(f"Chronic draw merchants \u2014 {draws} draws from {n} games suggests lack of winning mentality or clinical edge")

    # --- TOURNAMENT PEDIGREE ---
    tm = tourn.get('matches', 0)
    tw = (tourn.get('wins',0) + 0.5*tourn.get('draws',0)) / max(tm, 1) * 100
    if   tm >= 12 and tw >= 68:
        S.append(f"Exceptional tournament pedigree \u2014 {tw:.0f}% win rate in major tournaments over the last decade")
    elif tm >= 8  and tw >= 55:
        S.append(f"Strong major tournament record \u2014 handles pressure well ({tw:.0f}% win rate, {tm} games)")
    elif tm <= 3:
        W.append("Limited major tournament experience \u2014 may struggle with high-pressure knockout intensity")
    elif tm >= 8 and tw <= 35:
        W.append(f"Disappointing tournament history \u2014 only {tw:.0f}% win rate in big competitions over the last decade")

    # --- ELO ELITE TIER ---
    if   elo >= 2150: S.append(f"Among the highest-rated national teams ever \u2014 Elo rating ({round(elo)}) confirms world-class status")
    elif elo >= 2050: S.append(f"Top-tier Elo ({round(elo)}) \u2014 consistently among the world's best-performing nations")
    elif elo <= 1680: W.append(f"Low Elo ({round(elo)}) \u2014 significant quality gap to most opponents in this round")
    elif elo <= 1750: W.append(f"Below-average Elo ({round(elo)}) \u2014 historically struggles against world-class opposition")

    # --- SUBTLE/TACTICAL WEAKNESSES (For elite teams) ---
    if not W:
        if 0.88 < defense < 1.08: W.append(f"Occasional defensive lapses \u2014 can be punished by clinical opposition ({ga:.1f} g/game)")
        elif 0.92 < attack < 1.15: W.append("Reliant on defensive solidity \u2014 attack can sometimes look predictable against low blocks")
        elif cs < 35: W.append(f"Struggles to keep clean sheets ({cs}%) \u2014 heavily relies on the attack outscoring the opponent")
        elif 1750 < elo < 2000: W.append("High quality, but historically can struggle to consistently dominate the absolute elite tier")
        elif form < 75: W.append("Hasn't fully peaked \u2014 recent performances show room for improvement before hitting top gear")
        elif btts > 45: W.append(f"Games tend to be open ({btts}% BTTS) \u2014 can be drawn into chaotic matches")

    # Cap to most impactful items
    if not S: S.append("Balanced overall profile \u2014 no standout strengths but no glaring holes either")
    if not W: W.append("Tactically complete \u2014 an exceptionally well-rounded team with no obvious statistical vulnerabilities")
    return S[:6], W[:5]

# ============================================================
#  COMPOSITE RATING v2 (v8.0: includes momentum + knockout factor)
# ============================================================
def composite_rating(elo, attack, defense, form, tourn, fifa_pts=None, momentum_mult=1.0, ko_factor=1.0):
    elo_s  = min(100, max(0, (elo-1000)/12))
    att_s  = min(100, max(0, attack*50))
    def_s  = min(100, max(0, (2-defense)*50))
    tw = (tourn['wins']+0.5*tourn['draws'])/max(tourn['matches'],1)*100
    mom_s  = min(100, max(0, momentum_mult * 50))  # 1.0 = 50, 1.08 = 54
    ko_s   = min(100, max(0, ko_factor * 50))       # 1.0 = 50, 1.08 = 54
    
    # If FIFA points available, blend them in
    if fifa_pts:
        fifa_s = min(100, max(0, (fifa_pts-900)/12))
        return round(elo_s*0.22 + fifa_s*0.12 + att_s*0.18 + def_s*0.18 + form*0.10 + tw*0.08 + mom_s*0.06 + ko_s*0.06, 1)
    
    return round(elo_s*0.28 + att_s*0.18 + def_s*0.18 + form*0.12 + tw*0.08 + mom_s*0.08 + ko_s*0.08, 1)

# ============================================================
#  v8.0 ADVANCED SIMULATION ENGINE
#  - Negative Binomial distribution (overdispersion)
#  - Venue/travel factor
#  - Momentum multiplier
#  - Knockout pressure factor
#  - Rest day fatigue factor
#  - Dixon-Coles correction + H2H boost
# ============================================================
def simulate_match(ds1, ds2, disp1, disp2, strengths, elo_ratings, global_avg, form_scores,
                   h2h=None, momentum1=None, momentum2=None, ko1=None, ko2=None,
                   rest_days1=None, rest_days2=None, target_date=None):
    s1 = strengths.get(ds1, {'attack':1.0,'defense':1.0})
    s2 = strengths.get(ds2, {'attack':1.0,'defense':1.0})
    e1, e2 = elo_ratings.get(ds1,1500), elo_ratings.get(ds2,1500)
    f1 = form_scores.get(ds1, 50)/100
    f2 = form_scores.get(ds2, 50)/100

    # Base expected goals from Dixon-Coles style model
    elo_adj1 = 1 + (e1-e2)/4000
    elo_adj2 = 1 - (e1-e2)/4000
    form_adj1 = 0.88 + 0.24*f1
    form_adj2 = 0.88 + 0.24*f2

    # H2H goal boost: v8.0 time-weighted recent encounters influence predictions heavily!
    h2h_gf1 = h2h.get('weighted_t1_gf', h2h.get('t1_gf', 0)) if h2h else 0
    h2h_gf2 = h2h.get('weighted_t1_ga', h2h.get('t1_ga', 0)) if h2h else 0
    h2h_n   = max(h2h.get('weighted_total', h2h.get('total', 1)), 0.01) if h2h else 1
    h2h_avg1 = h2h_gf1 / h2h_n
    h2h_avg2 = h2h_gf2 / h2h_n
    h2h_boost1 = max(0.9, min(1.2, h2h_avg1 / max(global_avg, 0.5))) if h2h and h2h.get('total', 0)>=3 else 1.0
    h2h_boost2 = max(0.9, min(1.2, h2h_avg2 / max(global_avg, 0.5))) if h2h and h2h.get('total', 0)>=3 else 1.0

    # v8.0: Recent encounter direct winner nudge (within 180 days)
    recent_h2h_nudge1 = 1.0
    recent_h2h_nudge2 = 1.0
    if h2h and h2h.get('meetings') and target_date:
        latest_meet = h2h['meetings'][0]
        try:
            meet_date = pd.to_datetime(latest_meet['date'])
            days_ago = (target_date - meet_date).days
            if 0 <= days_ago <= 180:
                # Nudge decays over a 60-day half-life
                recency_weight = math.exp(-math.log(2) * days_ago / 60.0)
                if latest_meet['result'] == 'W':
                    recent_h2h_nudge1 += 0.15 * recency_weight
                    recent_h2h_nudge2 -= 0.08 * recency_weight
                elif latest_meet['result'] == 'L':
                    recent_h2h_nudge1 -= 0.08 * recency_weight
                    recent_h2h_nudge2 += 0.15 * recency_weight
        except Exception:
            pass

    # v8.0: Venue/Travel factor (replaces old hard-coded host boost)
    venue1 = compute_venue_factor(ds1)
    venue2 = compute_venue_factor(ds2)

    # v8.0: Momentum multiplier
    mom1 = momentum1['multiplier'] if momentum1 else 1.0
    mom2 = momentum2['multiplier'] if momentum2 else 1.0

    # v8.0: Knockout pressure factor
    kof1 = ko1['factor'] if ko1 else 1.0
    kof2 = ko2['factor'] if ko2 else 1.0

    # v8.0: Rest/fatigue factor (optimal rest = 4-5 days)
    def rest_factor(days):
        if days is None: return 1.0
        if days >= 5: return 1.0
        if days >= 4: return 0.99
        if days >= 3: return 0.97
        if days >= 2: return 0.94
        return 0.90  # played yesterday — extremely fatigued

    rf1 = rest_factor(rest_days1)
    rf2 = rest_factor(rest_days2)

    l1 = max(0.3, s1['attack'] * s2['defense'] * global_avg * elo_adj1 * form_adj1
             * GOAL_INFLATE * h2h_boost1 * venue1 * mom1 * kof1 * rf1 * recent_h2h_nudge1)
    l2 = max(0.3, s2['attack'] * s1['defense'] * global_avg * elo_adj2 * form_adj2
             * GOAL_INFLATE * h2h_boost2 * venue2 * mom2 * kof2 * rf2 * recent_h2h_nudge2)

    # Dixon-Coles low-score correction
    def dc_correction(i, j, mu1, mu2, rho=-0.1):
        if i==0 and j==0: return 1 - mu1*mu2*rho
        if i==1 and j==0: return 1 + mu2*rho
        if i==0 and j==1: return 1 + mu1*rho
        if i==1 and j==1: return 1 - rho
        return 1.0
        
    # Bivariate Urgency: open games breed more goals
    def bivariate_urgency(g1, g2):
        if g1 > 0 and g2 > 0:
            return 1.0 + 0.08 * min(g1, g2)
        return 1.0

    # v8.0: Negative Binomial PMF (captures overdispersion in football scores)
    # NB parameterized by n (number of successes) and p (probability)
    # Mean = n*(1-p)/p, so p = n/(n+lambda), where n = NBINOM_R
    def nb_pmf(k, lam):
        r = NBINOM_R
        p = r / (r + lam)
        return nbinom.pmf(k, r, p)

    MAXG = 12
    p1w = p2w = pd_ = 0.0
    score_probs = {}
    for g1 in range(MAXG+1):
        for g2 in range(MAXG+1):
            p = nb_pmf(g1,l1) * nb_pmf(g2,l2) * dc_correction(g1,g2,l1,l2) * bivariate_urgency(g1,g2)
            score_probs[(g1,g2)] = max(0,p)

    # Normalize
    total_p = sum(score_probs.values())
    score_probs = {k:v/total_p for k,v in score_probs.items()}

    for (g1,g2),p in score_probs.items():
        if g1>g2: p1w+=p
        elif g1<g2: p2w+=p
        else: pd_+=p

    xg1 = sum(g1*p for (g1,g2),p in score_probs.items())
    xg2 = sum(g2*p for (g1,g2),p in score_probs.items())
    top5 = sorted(score_probs.items(), key=lambda x:x[1], reverse=True)[:5]

    # Most likely score (this is ALWAYS the 90-minute prediction)
    ms1, ms2 = top5[0][0]
    
    if ms1 == ms2:
        # Draw after 90 minutes — does it go to ET or Pens?
        # Use extra time lambdas (~1/3 of 90 min match intensity)
        et_l1, et_l2 = l1 / 3.0, l2 / 3.0
        
        if et_l1 > et_l2 + 0.15:
            winner = disp1
            score_str = f"{ms1+1}-{ms2} ({winner} wins in ET)"
            outcome = "Extra Time"
        elif et_l2 > et_l1 + 0.15:
            winner = disp2
            score_str = f"{ms1}-{ms2+1} ({winner} wins in ET)"
            outcome = "Extra Time"
        else:
            pen_edge1 = p1w * kof1
            pen_edge2 = p2w * kof2
            winner = disp1 if pen_edge1 > pen_edge2 else disp2
            score_str = f"{ms1}-{ms2} ({winner} wins on Pens)"
            outcome = "Penalties"
    else:
        winner = disp1 if ms1 > ms2 else disp2
        score_str = f"{ms1}-{ms2}"
        outcome = "Normal Time"

    return {
        'lambda1':         round(l1,3),
        'lambda2':         round(l2,3),
        'xg1':             round(xg1,2),
        'xg2':             round(xg2,2),
        'prob_t1_win':     round(p1w*100,1),
        'prob_t2_win':     round(p2w*100,1),
        'prob_draw':       round(pd_*100,1),
        'predicted_score': score_str,
        'predicted_outcome': outcome,
        'winner':          winner,
        'winner_ds':       ds(winner),
        'elo_t1':          round(e1),
        'elo_t2':          round(e2),
        'top_scores':      [{'score':f"{g1}-{g2}",'pct':round(p*100,2)} for (g1,g2),p in top5],
    }

# ============================================================
#  BUILD TEAM PROFILE
# ============================================================
def build_profile(display, df, strengths, elo_ratings, target_date, fifa_rankings=None):
    if fifa_rankings is None: fifa_rankings = {}
    ds_name = ds(display)
    elo = elo_ratings.get(ds_name, 1500)
    st = strengths.get(ds_name, {'attack':1.0,'defense':1.0,'avg_gf':0,'avg_ga':0})
    form_score, form_results, last_game = compute_form(df, ds_name, target_date)
    gs = compute_goal_stats(df, ds_name, target_date)
    tourn = compute_tourn_record(df, ds_name, target_date)
    momentum = compute_momentum(df, ds_name, target_date)
    ko_factor = compute_knockout_factor(df, ds_name, target_date)
    venue_factor = compute_venue_factor(ds_name)
    S, W = swlabels(st['attack'], st['defense'], form_score, gs, tourn, elo, form_results)
    
    fifa_data = fifa_rankings.get(ds_name, {})
    fifa_pts = fifa_data.get('points')
    comp = composite_rating(elo, st['attack'], st['defense'], form_score, tourn, fifa_pts,
                            momentum['multiplier'], ko_factor['factor'])
    
    return {
        'display': display,
        'ds_name': ds_name,
        'elo':         round(elo),
        'fifa_rank':   fifa_data.get('rank'),
        'composite':   comp,
        'attack':      st['attack'],
        'defense':     st['defense'],
        'avg_gf':      st.get('avg_gf',0),
        'avg_ga':      st.get('avg_ga',0),
        'form_score':  form_score,
        'form_results':form_results,
        'last_game':   last_game,
        'goal_stats':  gs,
        'tourn_record':tourn,
        'momentum':    momentum,
        'knockout_factor': ko_factor,
        'venue_factor': venue_factor,
        'strengths':   S,
        'weaknesses':  W,
    }

def simulate_tournament(games, df, strengths, elo_ratings, global_avg, profiles, live_scores, target_date, quiet=False):
    round_types = ['r32', 'r16', 'qf', 'sf', 'final']
    round_name_map = {
        'r32': "Round of 32",
        'r16': "Round of 16",
        'qf': "Quarter-finals",
        'sf': "Semi-finals",
        'final': "Final"
    }

    form_scores = {p['ds_name']: p['form_score'] for p in profiles.values()}
    momentum_data = {p['ds_name']: p.get('momentum', {'multiplier': 1.0}) for p in profiles.values()}
    ko_data = {p['ds_name']: p.get('knockout_factor', {'factor': 1.0}) for p in profiles.values()}
    last_match_date = {}  # ds_name -> date of last match in this tournament

    ko_games = {}
    for g in games:
        gtype = g.get('type', '')
        if gtype not in round_types:
            continue
        gid = str(g.get('id', ''))
        if not gid:
            continue

        status = (g.get('status') or g.get('matchStatus') or g.get('time_elapsed') or str(g.get('finished'))).upper()
        if status == 'TRUE': status = 'FINISHED'
        is_finished = any(w in status for w in ['FINISH','COMPLET','ENDED','FULL','FT'])
        
        hs = g.get('homeScore') or g.get('home_score') or g.get('homeGoals') or g.get('score', {}).get('home', None)
        as_ = g.get('awayScore') or g.get('away_score') or g.get('awayGoals') or g.get('score', {}).get('away', None)
        score_available = (hs not in [None, 'null', 'NULL', ''] and as_ not in [None, 'null', 'NULL', ''])

        try:
            hs_int = int(hs) if score_available else None
            as_int = int(as_) if score_available else None
        except (ValueError, TypeError):
            hs_int = None
            as_int = None
            score_available = False

        hp = g.get('home_penalty_score')
        ap = g.get('away_penalty_score')
        hp_int = None
        ap_int = None
        if hp not in [None, 'null', 'NULL', '']:
            try: hp_int = int(hp)
            except ValueError: pass
        if ap not in [None, 'null', 'NULL', '']:
            try: ap_int = int(ap)
            except ValueError: pass

        kickoff_str = g.get('local_date', '')
        try:
            dt = datetime.datetime.strptime(kickoff_str, "%m/%d/%Y %H:%M")
            kickoff_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            kickoff_iso = kickoff_str

        ko_games[gid] = {
            'id': gid,
            'type': gtype,
            'home_label': g.get('home_team_label'),
            'away_label': g.get('away_team_label'),
            'home_team': map_team_name(g.get('home_team_name_en')),
            'away_team': map_team_name(g.get('away_team_name_en')),
            'finished': is_finished and score_available,
            'home_score': hs_int,
            'away_score': as_int,
            'home_pen': hp_int,
            'away_pen': ap_int,
            'kickoff': kickoff_iso,
            'winner': None
        }

    def resolve_team_from_label(label):
        if not label:
            return None
        if label.startswith("Winner Match "):
            prev_id = label.replace("Winner Match ", "").strip()
            prev_match = ko_games.get(prev_id)
            if prev_match:
                return prev_match.get('winner')
        return None

    all_rounds = []
    for gtype in round_types:
        rnd_name = round_name_map[gtype]
        if not quiet:
            print(f"\n--- {rnd_name} ---")
        
        rnd_matches = [m for m in ko_games.values() if m['type'] == gtype]
        rnd_matches.sort(key=lambda x: int(x['id']))

        round_matches = []
        for m_info in rnd_matches:
            gid = m_info['id']
            t1 = m_info['home_team'] or resolve_team_from_label(m_info['home_label'])
            t2 = m_info['away_team'] or resolve_team_from_label(m_info['away_label'])

            if not t1: t1 = "TBD"
            if not t2: t2 = "TBD"

            t1d, t2d = t1, t2
            ds1, ds2 = ds(t1), ds(t2)

            h2h = compute_h2h(df, ds1, ds2, target_date)
            rest1 = (target_date - last_match_date[ds1]).days if ds1 in last_match_date else None
            rest2 = (target_date - last_match_date[ds2]).days if ds2 in last_match_date else None

            mom1 = momentum_data.get(ds1, {'multiplier': 1.0})
            mom2 = momentum_data.get(ds2, {'multiplier': 1.0})
            kof1 = ko_data.get(ds1, {'factor': 1.0})
            kof2 = ko_data.get(ds2, {'factor': 1.0})

            if m_info['finished']:
                real_home = m_info['home_score']
                real_away = m_info['away_score']
                t1_pen = m_info['home_pen']
                t2_pen = m_info['away_pen']

                if real_home > real_away:
                    winner = t1
                    real_str = f"{real_home}-{real_away}"
                elif real_home < real_away:
                    winner = t2
                    real_str = f"{real_home}-{real_away}"
                else:
                    if t1_pen is not None and t2_pen is not None:
                        if t1_pen > t2_pen:
                            winner = t1
                            real_str = f"{real_home}-{real_away} ({t1} wins on Pens)"
                        elif t2_pen > t1_pen:
                            winner = t2
                            real_str = f"{real_home}-{real_away} ({t2} wins on Pens)"
                        else:
                            winner = t1 if elo_ratings.get(ds1,0) > elo_ratings.get(ds2,0) else t2
                            real_str = f"{real_home}-{real_away}"
                    else:
                        winner = t1 if elo_ratings.get(ds1,0) > elo_ratings.get(ds2,0) else t2
                        real_str = f"{real_home}-{real_away}"

                sim = simulate_match(ds1, ds2, t1d, t2d, strengths, elo_ratings, global_avg, form_scores,
                                     h2h, mom1, mom2, kof1, kof2, rest1, rest2, target_date)
                match = {
                    'team1': t1d, 'team2': t2d, 'round': rnd_name,
                    'match_id': f"{t1d.replace(' ','-')}-vs-{t2d.replace(' ','-')}",
                    'real_score': real_str,
                    'real_winner': winner,
                    'predicted_score': sim['predicted_score'],
                    'predicted_winner': sim['winner'],
                    'winner': winner,
                    'is_played': True,
                    'kickoff': m_info['kickoff'],
                    **{k:v for k,v in sim.items() if k not in ('predicted_score','winner')},
                    'h2h': h2h,
                    't1_profile': profiles.get(t1d,{}),
                    't2_profile': profiles.get(t2d,{}),
                }
                if not quiet:
                    print(f"  {t1d} [{real_home}-{real_away}] {t2d}  (PLAYED) Winner: {winner}")
            else:
                sim = simulate_match(ds1, ds2, t1d, t2d, strengths, elo_ratings, global_avg, form_scores,
                                     h2h, mom1, mom2, kof1, kof2, rest1, rest2, target_date)
                match = {
                    'team1': t1d, 'team2': t2d, 'round': rnd_name,
                    'match_id': f"{t1d.replace(' ','-')}-vs-{t2d.replace(' ','-')}",
                    'real_score': None,
                    'real_winner': None,
                    'is_played': False,
                    'kickoff': m_info['kickoff'],
                    **sim,
                    'h2h': h2h,
                    't1_profile': profiles.get(t1d,{}),
                    't2_profile': profiles.get(t2d,{}),
                }
                if not quiet:
                    print(f"  {t1d} {sim['predicted_score']} {t2d}  ->  {sim['winner']}  ({sim['prob_t1_win']}% / {sim['prob_t2_win']}%)")
                winner = sim['winner']

            last_match_date[ds1] = target_date
            last_match_date[ds2] = target_date

            ko_games[gid]['winner'] = winner
            round_matches.append(match)
        
        all_rounds.append({'round': rnd_name, 'matches': round_matches})

    return all_rounds

# ============================================================
#  VERSIONED OUTPUT SAVING
# ============================================================
def save_output(output, out_dir):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join(out_dir, 'outputs'), exist_ok=True)

    json_str = json.dumps(output, ensure_ascii=False, default=str)
    js_payload = f"const PREDICTOR_DATA = {json_str};"

    # 1) Save timestamped snapshot
    snap_path = os.path.join(out_dir, 'outputs', f'data_{ts}.js')
    with open(snap_path, 'w', encoding='utf-8') as f:
        f.write(js_payload)
    print(f"Snapshot saved: {snap_path}")

    # 2) Overwrite data.js (latest)
    latest_path = os.path.join(out_dir, 'data.js')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(js_payload)
    print(f"Latest data.js updated")

    # 3) Update runs_index.js
    index_path = os.path.join(out_dir, 'runs_index.js')
    existing_runs = []
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            raw = f.read().replace('const RUNS_INDEX = ','').strip().rstrip(';')
            try: existing_runs = json.loads(raw)
            except: existing_runs = []

    new_run = {
        'timestamp': ts,
        'file': f'outputs/data_{ts}.js',
        'champion': output['meta']['champion'],
        'live_scores_used': output['meta']['live_scores_used'],
        'matches_played': output['meta']['matches_played'],
        'generated_at': output['meta']['generated_at'],
    }
    existing_runs.insert(0, new_run)

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('const RUNS_INDEX = ' + json.dumps(existing_runs, ensure_ascii=False) + ';')
    print(f"runs_index.js updated ({len(existing_runs)} total runs)")

    return ts

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup 2026 Super Engine")
    parser.add_argument('--quiet', action='store_true', help="Suppress detailed output")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    # Fetch live tournament schedule/games (single network request)
    games = fetch_wc2026_games()

    # Load data
    df = load_data(games)
    elo_ratings = compute_elo(df, TARGET_DATE)
    strengths, global_avg = compute_strengths(df, TARGET_DATE)

    # Fetch live scores from games list
    live_scores = fetch_live_scores(games)
    
    # Fetch FIFA Rankings
    fifa_rankings = fetch_fifa_rankings()

    # Build team profiles dynamically from the R32 matches present in the API
    if not args.quiet:
        print("\nBuilding team profiles...")
    all_teams = set()
    for g in games:
        if g.get('type') == 'r32':
            t1 = map_team_name(g.get('home_team_name_en'))
            t2 = map_team_name(g.get('away_team_name_en'))
            if t1: all_teams.add(t1)
            if t2: all_teams.add(t2)
    all_teams = sorted(list(all_teams))

    profiles = {}
    for team in all_teams:
        profiles[team] = build_profile(team, df, strengths, elo_ratings, TARGET_DATE, fifa_rankings)
        p = profiles[team]
        if not args.quiet:
            print(f"  {team:28}  ELO:{p['elo']}  Form:{p['form_score']}%  Composite:{p['composite']}")

    # Simulate tournament dynamically from API bracket structure
    tournament_rounds = simulate_tournament(games, df, strengths, elo_ratings, global_avg, profiles, live_scores, TARGET_DATE, args.quiet)
    champion = tournament_rounds[-1]['matches'][0]['winner']

    matches_played = sum(1 for r in tournament_rounds for m in r['matches'] if m.get('is_played'))

    output = {
        'meta': {
            'generated_at':      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'champion':          champion,
            'total_teams':       len(all_teams),
            'data_source_hist':  'martj42/international_results',
            'data_source_live':  'worldcup26.ir',
            'hist_last_date':    df['date'].max().strftime('%Y-%m-%d'),
            'model':             'v8.0: NegBinom + Dixon-Coles + Opp-Quality + Momentum + KO Pressure + Venue/Travel + Fatigue',
            'half_life_days':    HALF_LIFE_DAYS,
            'goal_inflate':      GOAL_INFLATE,
            'live_scores_used':  len(live_scores) > 0,
            'matches_played':    matches_played,
        },
        'teams':  profiles,
        'rounds': tournament_rounds,
    }

    ts = save_output(output, out_dir)

    print(f"\nPredicted World Champion: {champion}")
    print(f"Snapshot timestamp: {ts}")
    print("Open index.html in your browser!")
