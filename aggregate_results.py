import csv
import os
import statistics

def is_numeric(val):
    try:
        val = val.replace('%', '')
        float(val)
        return True
    except ValueError:
        return False

def get_numeric(val):
    val = val.replace('%', '')
    return float(val)

def process_files(files, seeds_folders, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for filename in files:
        data = []
        # read all files
        for seed in seeds_folders:
            filepath = os.path.join(seed, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                data.append(list(reader))
                
        # Assume all seeds have same structure
        out_data = []
        base = data[0]
        
        for r in range(len(base)):
            row_out = []
            for c in range(len(base[r])):
                vals = []
                is_pct = False
                all_numeric = True
                
                for s in range(len(seeds_folders)):
                    val = data[s][r][c]
                    if '%' in val:
                        is_pct = True
                    if is_numeric(val):
                        vals.append(get_numeric(val))
                    else:
                        all_numeric = False
                        break
                        
                if all_numeric and len(vals) > 1 and r > 0: # r > 0 to skip header if it happens to be numeric
                    mean = statistics.mean(vals)
                    std = statistics.stdev(vals)
                    if std == 0:
                        row_out.append(base[r][c])
                    else:
                        fmt = f"{mean:.2f} ± {std:.2f}"
                        if is_pct:
                            fmt += "%"
                        row_out.append(fmt)
                else:
                    # use the first seed's value for non-numeric or header
                    row_out.append(base[r][c])
            out_data.append(row_out)
            
        out_filepath = os.path.join(out_folder, filename)
        with open(out_filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(out_data)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
seeds = [os.path.join(base_dir, "41"), os.path.join(base_dir, "42"), os.path.join(base_dir, "43")]
files = ["results.csv", "ototuss_tuning_results.csv", "ototuss_creative_tuning_results.csv", "creative_writing_results.csv"]
out_dir = os.path.join(base_dir, "final")

process_files(files, seeds, out_dir)
print(f"Aggregated files created in {out_dir}")
