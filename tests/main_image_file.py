if __name__ == '__main__':
    import sys
    from neurocarto.config import parse_cli
    from neurocarto.main_app import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=file',
    ]))
