import argparse
import traceback
import numpy as np
import tensorflow as tf
import pathlib

def save_tag_to_csv(fn, tag='test_metric'):
    parts = fn.parts
    parent = pathlib.Path(*parts[0:-2])
    subfolder_name = parts[-2]

    output_fn = '{}/{}_{}.csv'.format(parent, subfolder_name, tag.replace('/', '_'))
    print(f"Will save to {output_fn}")

    wall_step_values = []
    for e in tf.compat.v1.train.summary_iterator(str(fn)):
        for v in e.summary.value:
            if v.tag == tag:
                wall_step_values.append((e.wall_time, e.step, v.simple_value))
    np.savetxt(output_fn, wall_step_values, delimiter=',', fmt='%10.5f', header=f"wall_time,step,{tag}", comments='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=".")
    parser.add_argument('--tags', nargs='*', default=["experience_generation/episode_rew", "experience_generation/episode_rew_var", "experience_generation/episode_rew_std"])
    args = parser.parse_args()

    root = pathlib.Path(args.path)
    filenames = root.rglob("event*")

    for filename in filenames:
        try:
            print(f"Processing: {filename}")
            for tag in args.tags:
                save_tag_to_csv(filename, tag=tag)
        except Exception:
            print(traceback.format_exc())
