cat 'G:/My Drive/pypi_token.txt'
rm -r dist
python -m build
python -m twine upload --repository dist/*