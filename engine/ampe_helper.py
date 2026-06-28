import json, os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
INDEX_FILE = os.path.join(DATA_DIR, 'runs_index.js')

def load_index():
    if not os.path.exists(INDEX_FILE): return []
    try:
        raw = open(INDEX_FILE, encoding='utf-8').read().replace('const RUNS_INDEX = ','').strip().rstrip(';')
        return json.loads(raw)
    except:
        return []

def save_index(runs):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        f.write('const RUNS_INDEX = ' + json.dumps(runs, ensure_ascii=False) + ';')

def view():
    runs = load_index()
    if not runs:
        print("No saves found.")
        return
    for i, r in enumerate(runs):
        live_str = 'Live' if r.get('live_scores_used') else 'Prediction'
        champ = r.get('champion', 'Unknown')
        gen = r.get('generated_at', 'Unknown Date')
        print(f"[{i+1}] {gen}  |  Champion: {champ:15}  |  Mode: {live_str}")

def delete(idx):
    runs = load_index()
    if idx < 1 or idx > len(runs):
        print("Invalid selection.")
        return
    target = runs.pop(idx-1)
    # the file path in JSON is "outputs/data_..." 
    # we need to prepend DATA_DIR to it
    rel_path = target.get('file')
    if rel_path:
        file_path = os.path.join(DATA_DIR, rel_path)
        if os.path.exists(file_path):
            os.remove(file_path)
    save_index(runs)
    print(f"Deleted save from {target.get('generated_at')}.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'view':
            view()
        elif cmd == 'delete':
            try:
                delete(int(sys.argv[2]))
            except ValueError:
                print("Invalid number.")
