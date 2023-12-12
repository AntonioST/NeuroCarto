from bokeh.models import Div


def bold(text: str) -> Div:
    return Div(text=f'<b>{text}</b>')
