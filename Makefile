check:
	uv run ruff format .
	uv run ruff check . --fix

test:
	COVERAGE_FILE=.coverage pytest --cov=. --cov-report=term-missing --cov-report=xml
	@coverage=$$(python -c "import xml.etree.ElementTree as ET; r=ET.parse('coverage.xml').getroot(); print(f\"{float(r.attrib['line-rate'])*100:.1f}%\")"); \
	sed -i "" "s/Coverage: .*/Coverage: $$coverage/" README.md