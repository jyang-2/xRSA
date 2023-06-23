def luminance(c):
    lum = c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114
    return lum


def best_text_color(c):
    if c > 186 / 255:
        text_color = 'white'
    else:
        text_color = 'black'
