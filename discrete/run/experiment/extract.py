import argparse
import traceback
import numpy as np
import tensorflow as tf
import pathlib

def save_metrics_to_csv(fn, tags):
    parts = fn.parts
    parent = pathlib.Path(*parts[0:-2])
    subfolder_name = parts[-2]
    output_fn = '{}/{}_metrics.csv'.format(parent, subfolder_name)
    print(f"Will save to {output_fn}")

    
    data = {}

    for e in tf.compat.v1.train.summary_iterator(str(fn)):
        step = e.step
        for v in e.summary.value:
            if v.tag in tags:
                if step not in data:
                    data[step] = {'wall_time': e.wall_time}
                key = v.tag.split('/')[-1]
                data[step][key] = v.simple_value

    simplified_tags = [tag.split('/')[-1] for tag in tags]
    header = ['step', 'wall_time'] + simplified_tags

    rows = []
    for step in sorted(data.keys()):
        if any(tag in data[step] for tag in simplified_tags):
            row = [step, data[step].get('wall_time', 0)]
            for t in simplified_tags:
                row.append(data[step].get(t, np.nan))
            rows.append(row)

    # Save the rows to CSV.
    np.savetxt(output_fn, rows, delimiter=',', fmt='%10.5f', header=','.join(header), comments='')
    print(f"Saved metrics for {fn}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=".")
    parser.add_argument(
        '--tags',
        nargs='*',
        default=[
            "experience_generation/episode_rew",
            "experience_generation/episode_rew_var",
            "experience_generation/episode_rew_std"
        ]
    )
    args = parser.parse_args()

    root = pathlib.Path(args.path)
    filenames = root.rglob("event*")

    for filename in filenames:
        try:
            print(f"Processing: {filename}")
            save_metrics_to_csv(filename, args.tags)
        except Exception:
            print(traceback.format_exc())
