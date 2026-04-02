.PHONY: lint test check ci golden bench

lint:
	ruff check nexus/ && mypy nexus/

test:
	pytest tests/unit/

check: lint test

ci: lint test
	pytest tests/integration/
	bandit -r nexus/ -q
	safety check

golden:
	pytest tests/golden/

bench:
	pytest tests/benchmarks/
