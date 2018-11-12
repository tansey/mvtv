import matplotlib.pylab as plt
import numpy as np


if __name__ == '__main__':
    models = ['cart', 'crisp', 'gapcrisp', 'gfl']
    patches = [np.load('data/crime/austin/patches/{}_7patches.npy'.format(m)) for m in models]
    truth_patches = np.load('data/crime/austin/patches/truth_7patches.npy')

    p = np.random.randint(0, len(truth_patches))

    print ''
    for model, model_patches in zip(models, patches):
        patch = model_patches[p]
        print '-' * 50
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[0]]) + ' |'
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[1]]) + ' |'
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[2]]) + ' | '
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[3,:3]]) \
            + ' |      | ' \
            + ' | '.join(['{:.2f}'.format(x) for x in patch[3,4:]]) \
            + ' |'
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[4]]) + ' |'
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[5]]) + ' |'
        print '| ' + ' | '.join(['{:.2f}'.format(x) for x in patch[6]]) + ' |'
        print '-' * 50
        print ''
        print ''

    guess = float(raw_input('Guess? '))
    truth = truth_patches[p][3,3]
    print 'Truth: {} Your error: {}'.format(truth, np.sqrt((truth-guess)**2))