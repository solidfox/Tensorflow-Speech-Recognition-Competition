
def generate_result(model):
    sub_dir = "/submission/"
    for t in tqdm(it):
        fname, label = t['sample'].decode(), id2name[t['label']]
    submission[fname] = label
    with open(os.path.join(sub_dir, 'submission.csv'), 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write('{},{}\n'.format(fname, label))