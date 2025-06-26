from matplotlib import pyplot as plt

from Preprocessing import load_dataset


def plot_data(X, y, num_examples, categories, title, grayscale=True):
    num_categories = len(categories)
    for category in range(num_categories):
        i = 0
        # plot the first "num_examples" data of each category
        for col in range(num_examples):
            while y[i] != category:
                i = i + 1
            ax = plt.subplot(num_categories, num_examples, category * num_examples + col + 1)
            if grayscale:
                plt.imshow(X[i].squeeze(), cmap='gray')
            else:
                plt.imshow(X[i])
            if col == 0:
                ax.set_ylabel(categories[category])
            ax.set_yticks([])
            ax.get_xaxis().set_visible(False)
            i = i + 1
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(title)
    plt.show()


if __name__ == '__main__':
    grayscale = True
    X, _, _, y, _, _ = load_dataset(base_path='./dataset', grayscale=grayscale, data_augmentation=False)

    categories = ['Ruy Lopez', 'Italian Game', 'Queen\'s Gambit', 'Sicilian Defense', 'Nimzo-Indian Defense']
    plot_data(X, y, num_examples=5, categories=categories, title='Chess Openings Images', grayscale=grayscale)
