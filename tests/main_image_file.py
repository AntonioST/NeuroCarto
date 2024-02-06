if __name__ == '__main__':
    import sys
    from chmap.config import parse_cli
    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=file',
    ]))
