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
from scipy.stats import poisson
import numpy as np
warnings.filterwarnings('ignore')

# ============================================================
#  AGNEL MATCH PREDICTION ENGINE (AMPE) v7.0
#  - Live scores from worldcup26.ir (free, no auth needed)
#  - WC2026 GROUP STAGE results injected into training data
#  - Deep football-analyst strengths/weaknesses (12+ dimensions)
#  - Versioned outputs + runs_index.js
#  - Dixon-Coles + H2H boost + Elo + form scoring + FIFA ranks
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
    "Congo DR":              ["DR Congo","Congo DR","Congo-Kinshasa","Democratic Republic of Congo"],
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

# ============================================================
#  LIVE SCORES from worldcup26.ir
# ============================================================
def fetch_live_scores():
    """Fetch live/finished match results from worldcup26.ir API.
    Returns a dict keyed by frozenset of team display names -> {score, status, home, away}
    """
    print("Fetching live WC2026 scores from worldcup26.ir...")
    try:
        resp = requests.get(LIVE_API_URL, timeout=10)
        if resp.status_code != 200:
            print(f"  API returned status {resp.status_code}, skipping live scores")
            return {}
        data = resp.json()
        games = data if isinstance(data, list) else data.get('data', data.get('games', data.get('matches', [])))
    except Exception as e:
        print(f"  Could not fetch live scores: {e}")
        return {}

    live_scores = {}
    count = 0
    for game in games:
        # API field names can vary - try multiple
        home = (game.get('homeTeam') or game.get('home_team') or game.get('home') or {})
        away = (game.get('awayTeam') or game.get('away_team') or game.get('away') or {})
        if isinstance(home, str): home_name = home
        else: home_name = home.get('name') or home.get('en') or home.get('nameEn') or ''
        if isinstance(away, str): away_name = away
        else: away_name = away.get('name') or away.get('en') or away.get('nameEn') or ''

        status = (game.get('status') or game.get('matchStatus') or '').upper()
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
            key = frozenset([mapped_home, mapped_away])
            is_finished = any(w in status for w in ['FINISH','COMPLET','ENDED','FULL','FT'])
            score_available = (hs is not None and as_ is not None)
            live_scores[key] = {
                'home': mapped_home,
                'away': mapped_away,
                'home_score': int(hs) if score_available else None,
                'away_score': int(as_) if score_available else None,
                'status': status,
                'finished': is_finished and score_available,
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

def fetch_wc2026_history():
    """Fetch all FINISHED WC2026 group stage matches from worldcup26.ir
    and return a DataFrame compatible with the martj42 dataset."""
    print("Fetching WC2026 completed matches from worldcup26.ir for training data...")
    try:
        resp = requests.get(LIVE_API_URL, timeout=12)
        if resp.status_code != 200:
            print(f"  API returned {resp.status_code}, skipping WC2026 history injection")
            return None
        raw = resp.json()
        games = raw if isinstance(raw, list) else raw.get('games', raw.get('data', []))
    except Exception as e:
        print(f"  Could not fetch WC2026 history: {e}")
        return None

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


def load_data():
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
    df_wc = fetch_wc2026_history()
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
#  ATTACK / DEFENSE STRENGTHS
# ============================================================
def compute_strengths(df, target_date):
    print("Computing attack/defense strengths (time-decay weighted)...")
    cutoff = target_date - pd.Timedelta(days=365*YEARS_BACK)
    dfw = df[(df['date']>=cutoff)&(df['date']<target_date)].copy()
    dfw['w'] = dfw.apply(lambda r: match_weight(r, target_date), axis=1)
    tw = dfw['w'].sum()
    global_avg = ((dfw['home_score']*dfw['w']).sum() + (dfw['away_score']*dfw['w']).sum()) / (2*tw)
    teams = set(dfw['home_team'])|set(dfw['away_team'])
    stats = {}
    for team in teams:
        hm = dfw[dfw['home_team']==team]
        am = dfw[dfw['away_team']==team]
        tw_ = hm['w'].sum()+am['w'].sum()
        if tw_ < 0.01:
            stats[team] = {'attack':1.0,'defense':1.0,'avg_gf':global_avg,'avg_ga':global_avg,'n':0}
            continue
        gf = (hm['home_score']*hm['w']).sum()+(am['away_score']*am['w']).sum()
        ga = (hm['away_score']*hm['w']).sum()+(am['home_score']*am['w']).sum()
        agf, aga = gf/tw_, ga/tw_
        stats[team] = {
            'attack':  round(agf/global_avg, 4) if global_avg>0 else 1.0,
            'defense': round(aga/global_avg, 4) if global_avg>0 else 1.0,
            'avg_gf':  round(agf, 3),
            'avg_ga':  round(aga, 3),
            'n':       len(hm)+len(am),
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
def compute_h2h(df, ds1, ds2):
    h2h = df[((df['home_team']==ds1)&(df['away_team']==ds2))|((df['home_team']==ds2)&(df['away_team']==ds1))].sort_values('date',ascending=False)
    t1w=t2w=draws=t1gf=t1ga=0
    meetings=[]
    for _,m in h2h.iterrows():
        ih=m['home_team']==ds1
        gf=m['home_score'] if ih else m['away_score']; gc=m['away_score'] if ih else m['home_score']
        t1gf+=gf; t1ga+=gc
        if gf>gc: t1w+=1; r='W'
        elif gf<gc: t2w+=1; r='L'
        else: draws+=1; r='D'
        meetings.append({'date':m['date'].strftime('%Y-%m-%d'),'score':f"{gf}-{gc}",'result':r,'tourn':str(m.get('tournament',''))})
    return {'t1_wins':t1w,'t2_wins':t2w,'draws':draws,'total':t1w+t2w+draws,'t1_gf':t1gf,'t1_ga':t1ga,'meetings':meetings[:10]}

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
#  COMPOSITE RATING
# ============================================================
def composite_rating(elo, attack, defense, form, tourn, fifa_pts=None):
    elo_s  = min(100, max(0, (elo-1000)/12))
    att_s  = min(100, max(0, attack*50))
    def_s  = min(100, max(0, (2-defense)*50))
    tw = (tourn['wins']+0.5*tourn['draws'])/max(tourn['matches'],1)*100
    
    # If FIFA points available, blend them in
    if fifa_pts:
        fifa_s = min(100, max(0, (fifa_pts-900)/12))
        return round(elo_s*0.25 + fifa_s*0.15 + att_s*0.20 + def_s*0.20 + form*0.10 + tw*0.10, 1)
    
    return round(elo_s*0.35 + att_s*0.20 + def_s*0.20 + form*0.15 + tw*0.10, 1)

# ============================================================
#  IMPROVED POISSON SIMULATION
#  - Uses GOAL_INFLATE to push lambdas higher
#  - Uses negative binomial dispersion parameter to spread scores wider
#  - H2H nudge: if teams historically score freely vs each other, boost lambda
# ============================================================
def simulate_match(ds1, ds2, disp1, disp2, strengths, elo_ratings, global_avg, form_scores, h2h=None):
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

    # H2H goal boost: if these teams historically score a lot together
    h2h_gf1 = h2h['t1_gf'] if h2h else 0
    h2h_gf2 = h2h['t1_ga'] if h2h else 0
    h2h_n   = max(h2h['total'], 1) if h2h else 1
    h2h_avg1 = h2h_gf1 / h2h_n
    h2h_avg2 = h2h_gf2 / h2h_n
    h2h_boost1 = max(0.9, min(1.2, h2h_avg1 / max(global_avg, 0.5))) if h2h and h2h['total']>=3 else 1.0
    h2h_boost2 = max(0.9, min(1.2, h2h_avg2 / max(global_avg, 0.5))) if h2h and h2h['total']>=3 else 1.0

    HOSTS = {"United States", "USA", "Canada", "Mexico"}
    CONCACAF = {"United States", "USA", "Canada", "Mexico", "Panama", "Costa Rica", "Jamaica", "Honduras"}
    
    host_boost1 = 1.15 if ds1 in HOSTS else (1.05 if ds1 in CONCACAF else 1.0)
    host_boost2 = 1.15 if ds2 in HOSTS else (1.05 if ds2 in CONCACAF else 1.0)

    l1 = max(0.3, s1['attack'] * s2['defense'] * global_avg * elo_adj1 * form_adj1 * GOAL_INFLATE * h2h_boost1 * host_boost1)
    l2 = max(0.3, s2['attack'] * s1['defense'] * global_avg * elo_adj2 * form_adj2 * GOAL_INFLATE * h2h_boost2 * host_boost2)

    # Dixon-Coles low-score correction (discourages 0-0 inflation, makes 1-0/0-1 less dominant)
    def dc_correction(i, j, mu1, mu2, rho=-0.1):
        if i==0 and j==0: return 1 - mu1*mu2*rho
        if i==1 and j==0: return 1 + mu2*rho
        if i==0 and j==1: return 1 + mu1*rho
        if i==1 and j==1: return 1 - rho
        return 1.0
        
    # Advanced Bivariate Urgency: higher scores increase variance/openness
    def bivariate_urgency(g1, g2):
        if g1 > 0 and g2 > 0:
            return 1.0 + 0.08 * min(g1, g2)
        return 1.0

    MAXG = 12
    p1w = p2w = pd_ = 0.0
    score_probs = {}
    for g1 in range(MAXG+1):
        for g2 in range(MAXG+1):
            p = poisson.pmf(g1,l1)*poisson.pmf(g2,l2)*dc_correction(g1,g2,l1,l2)*bivariate_urgency(g1,g2)
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

    # Most likely score (avoid draw in knockout)
    ms1, ms2 = top5[0][0]
    if ms1==ms2:
        non_draw = [(s,p) for (s,p) in top5 if s[0]!=s[1]]
        if non_draw:
            ms1, ms2 = non_draw[0][0]
        elif l1>l2: ms1+=1
        else: ms2+=1

    winner = disp1 if ms1>ms2 else disp2

    return {
        'lambda1':         round(l1,3),
        'lambda2':         round(l2,3),
        'xg1':             round(xg1,2),
        'xg2':             round(xg2,2),
        'prob_t1_win':     round(p1w*100,1),
        'prob_t2_win':     round(p2w*100,1),
        'prob_draw':       round(pd_*100,1),
        'predicted_score': f"{ms1}-{ms2}",
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
    S, W = swlabels(st['attack'], st['defense'], form_score, gs, tourn, elo, form_results)
    
    fifa_data = fifa_rankings.get(ds_name, {})
    fifa_pts = fifa_data.get('points')
    comp = composite_rating(elo, st['attack'], st['defense'], form_score, tourn, fifa_pts)
    
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
        'strengths':   S,
        'weaknesses':  W,
    }

# ============================================================
#  TOURNAMENT SIMULATION
# ============================================================
def simulate_tournament(fixtures, df, strengths, elo_ratings, global_avg, profiles, live_scores, target_date, quiet=False):
    rounds = ["Round of 32","Round of 16","Quarter-finals","Semi-finals","Final"]
    cur = fixtures
    all_rounds = []
    form_scores = {p['ds_name']: p['form_score'] for p in profiles.values()}

    for rnd in rounds:
        if not quiet:
            print(f"\n--- {rnd} ---")
        round_matches = []
        winners = []

        for t1d, t2d in cur:
            ds1, ds2 = ds(t1d), ds(t2d)
            h2h = compute_h2h(df, ds1, ds2)
            key = frozenset([t1d, t2d])
            live = live_scores.get(key)

            if live and live['finished']:
                # Use real score
                real_home = live['home_score'] if live['home']==t1d else live['away_score']
                real_away = live['away_score'] if live['home']==t1d else live['home_score']
                if real_home > real_away: winner = t1d
                elif real_home < real_away: winner = t2d
                else: winner = t1d if ds1 > ds2 else t2d  # tiebreak by elo
                # Also compute prediction for reference
                sim = simulate_match(ds1, ds2, t1d, t2d, strengths, elo_ratings, global_avg, form_scores, h2h)
                match = {
                    'team1': t1d, 'team2': t2d, 'round': rnd,
                    'match_id': f"{t1d.replace(' ','-')}-vs-{t2d.replace(' ','-')}",
                    'real_score': f"{real_home}-{real_away}",
                    'real_winner': winner,
                    'predicted_score': sim['predicted_score'],
                    'winner': winner,  # knockout uses real result
                    'is_played': True,
                    **{k:v for k,v in sim.items() if k not in ('predicted_score','winner')},
                    'h2h': h2h,
                    't1_profile': profiles.get(t1d,{}),
                    't2_profile': profiles.get(t2d,{}),
                }
                if not quiet:
                    print(f"  {t1d} [{real_home}-{real_away}] {t2d}  (PLAYED) Winner: {winner}")
            else:
                # Predict
                sim = simulate_match(ds1, ds2, t1d, t2d, strengths, elo_ratings, global_avg, form_scores, h2h)
                match = {
                    'team1': t1d, 'team2': t2d, 'round': rnd,
                    'match_id': f"{t1d.replace(' ','-')}-vs-{t2d.replace(' ','-')}",
                    'real_score': None,
                    'real_winner': None,
                    'is_played': False,
                    **sim,
                    'h2h': h2h,
                    't1_profile': profiles.get(t1d,{}),
                    't2_profile': profiles.get(t2d,{}),
                }
                if not quiet:
                    print(f"  {t1d} {sim['predicted_score']} {t2d}  ->  {sim['winner']}  ({sim['prob_t1_win']}% / {sim['prob_t2_win']}%)")
                winner = sim['winner']

            round_matches.append(match)
            winners.append(winner)

        all_rounds.append({'round': rnd, 'matches': round_matches})

        if rnd != "Final":
            next_f = []
            for i in range(0, len(winners)-1, 2):
                next_f.append((winners[i], winners[i+1]))
            cur = next_f

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

    # Load data
    df = load_data()
    elo_ratings = compute_elo(df, TARGET_DATE)
    strengths, global_avg = compute_strengths(df, TARGET_DATE)

    # Fetch live scores
    live_scores = fetch_live_scores()
    
    # Fetch FIFA Rankings
    fifa_rankings = fetch_fifa_rankings()

    # Build team profiles
    if not args.quiet:
        print("\nBuilding team profiles...")
    all_teams = sorted(set(t for pair in fixtures_ro32 for t in pair))
    profiles = {}
    for team in all_teams:
        profiles[team] = build_profile(team, df, strengths, elo_ratings, TARGET_DATE, fifa_rankings)
        p = profiles[team]
        if not args.quiet:
            print(f"  {team:28}  ELO:{p['elo']}  Form:{p['form_score']}%  Composite:{p['composite']}")

    # Simulate tournament
    tournament_rounds = simulate_tournament(fixtures_ro32, df, strengths, elo_ratings, global_avg, profiles, live_scores, TARGET_DATE, args.quiet)
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
            'model':             'Elo + Time-Weighted Poisson + Dixon-Coles Correction + H2H Boost',
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
