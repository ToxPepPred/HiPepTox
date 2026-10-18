
def comput_gem_fea(seq_id, seq):
    file_path = './outputs/pdb2rr/{}.pdb2rr'.format(seq_id)
    with open(file_path, 'r') as fdr:
        fdrlines = fdr.readlines()

    N = len(seq)

    countlist = [0 for _ in range(N)]
    for line in fdrlines:
        res1 = line.split()[0]
        res2 = line.split()[1]
        if((int(res1) > N) or (int(res2) > N)):
                continue
        countlist[int(res1)-1] += 1
        countlist[int(res2)-1] += 1
    fdr.close()
    for i in range(len(countlist)):
        if (countlist[i] > 25):
            countlist[i] = 1
        else:
            countlist[i] /= 25
    
    countlist = [[x] for x in countlist]

    return countlist




