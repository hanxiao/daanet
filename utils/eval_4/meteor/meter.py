import os


def compute_meter_score(pred, ref):
    cwd = os.path.dirname(__file__)
    test_path = '{}/test'.format(cwd)
    ref_path = '{}/reference'.format(cwd)
    jar_path = '{}/meteor-1.5.jar'.format(cwd)
    save_path = '{}/res.txt'.format(cwd)
    with open(test_path, 'w') as f:
        f.write('\n'.join(pred))
    with open(ref_path, 'w') as f:
        f.write('\n'.join(ref))
    os.system('java -Xmx2G -jar {} {} {} -l en -norm > {}'.format(jar_path, test_path, ref_path, save_path))
    try:
        score = open(save_path).read().split('\n')[-2]
        return float(score.split(' ')[-1])
    except:
        return 0.0
