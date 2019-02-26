import matplotlib.pyplot as plt

from helper import draw_sample
from preprocess import equally_spaced_points_patterns, make_patterns_fixed_size
from ujipen.ujipen_class import UJIPen


def test_equal_spaced(total_points=10):
    ujipen = UJIPen(force_read=True)
    samples = ujipen.get_samples()
    for word in samples.keys():
        trials_multiple_strokes = list(filter(lambda trial: len(trial) > 1, samples[word]))
        if len(trials_multiple_strokes) > 0:
            samples[word] = trials_multiple_strokes
    samples_dummy = make_patterns_fixed_size(samples, total_points=total_points)
    samples_equally_spaced = equally_spaced_points_patterns(samples, total_points=total_points)
    for word in samples.keys():
        plt.clf()

        plt.subplot(131)
        plt.title('Original')
        draw_sample(samples[word][0])

        plt.subplot(132)
        plt.title('Dummy linspace')
        draw_sample(samples_dummy[word][0])

        plt.subplot(133)
        plt.title('Equally spaced')
        draw_sample(samples_equally_spaced[word][0])

        plt.show()


if __name__ == '__main__':
    test_equal_spaced()
