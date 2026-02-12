import os
import warnings


def save_plot(figure, path):
    split = os.path.split(path)
    if split[0] != '':
        os.makedirs(split[0], exist_ok=True)
    # 忽略字体警告
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Glyph.*missing from font')
        figure.savefig(path, format='pdf')
    pass
