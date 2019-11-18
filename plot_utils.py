import matplotlib.pyplot as plt


def plot_pie_half_usage(labels, label_values, title, test_show_plot=False):
    # Create a pieplot
    colors = ['Grey', 'Purple', 'Blue', 'Green', 'Orange', 'Red', 'Magenta', 'White']
    # label_values.append(sum(label_values))
    # labels.append('White')
    print(len(labels), labels)
    print(len(label_values), label_values)
    plt.pie(label_values[0:7] + [label_values[-1]], colors=colors, counterclock=False, startangle=180)  # , labels=labels)

    # add a circle at the center
    my_circle = plt.Circle((0, 0), 0.55, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    plt.title(title, pad=-145)
    if test_show_plot:
        plt.show()
        return
    plt.savefig('figures/meld/fig_' + title.split('\n')[0])
    plt.close()


def plot_normal_bars(labels, label_values, title, test_show_plot=False):
    plt.rcParams.update({'font.size': 16})
    plt.bar(labels, label_values)
    plt.title(title.split('\n')[0] + ' - ' + title.split('\n')[1])
    plt.xticks(rotation=15)
    # plt.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.ylim(.5, 5.5)
    # plt.xlim(.5, 5.5)
    # plt.xlabel('Emotions')
    # plt.ylabel('Number of Utterances')
    if test_show_plot:
        plt.show()
        return
    plt.savefig('figures/meld/fig_' + title.split('\n')[0])
    plt.close()


# create data
# emotions = ['12', '11', '3', '30', '0', '23', '4']
# values = [12, 11, 3, 30, 0, 23, 4]
# title = 'tag\nHello World'
#
# plot_pie_half_usage(emotions, values, title, test_show_plot=True)
# plot_normal_bars(emotions, values, title, test_show_plot=True)
