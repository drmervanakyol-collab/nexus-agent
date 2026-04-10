.PHONY: lint test check ci ci-commit ci-pr ci-release coverage
.PHONY: golden bench property mutmut dead-code security

# ─── Lint ─────────────────────────────────────────────────────────────────────
lint:
	ruff check nexus/ tests/
	mypy nexus/

# ─── Individual test targets ──────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --timeout=60

property:
	pytest tests/property/ -v

golden:
	NEXUS_GOLDEN_TESTS=1 pytest tests/golden/ -v -s

bench:
	pytest tests/benchmarks/ -v --timeout=120

# ─── Coverage (combined unit + integration + property) ────────────────────────
coverage:
	pytest tests/unit/ -v --timeout=60 \
	  --cov=nexus --cov-report=json -q
	pytest tests/integration/ -v --timeout=60 \
	  --cov=nexus --cov-append --cov-report=json -q
	pytest tests/property/ -v \
	  --cov=nexus --cov-append --cov-report=term-missing --cov-report=json
	python scripts/check_coverage.py

# ─── Standalone security / dead-code ─────────────────────────────────────────
security:
	bandit -r nexus/ -ll -q
	safety check --full-report

dead-code:
	vulture nexus/ --min-confidence 80

# ─── Mutation testing ─────────────────────────────────────────────────────────
mutmut:
	python scripts/run_mutation.py \
	  --paths nexus/core/ \
	  --tests tests/property/ tests/integration/ \
	  --threshold 0.70

# ─── CI profiles ──────────────────────────────────────────────────────────────
# Every commit: lint + unit (80% global floor)
ci-commit: lint
	pytest tests/unit/ -v --timeout=60 \
	  --cov=nexus --cov-report=term-missing --cov-fail-under=80

# PR / merge: stages 1-5 with combined coverage check
ci-pr: lint
	@echo "--- Stage 2: Unit Tests ---"
	pytest tests/unit/ -v --timeout=60 \
	  --cov=nexus --cov-report=json --cov-fail-under=80
	@echo "--- Stage 3: Integration Tests ---"
	pytest tests/integration/ -v --timeout=60 \
	  --cov=nexus --cov-append --cov-report=json
	@echo "--- Stage 4: Property Tests ---"
	pytest tests/property/ -v \
	  --cov=nexus --cov-append --cov-report=json
	@echo "--- Stage 5: Adversarial Tests ---"
	pytest tests/adversarial/ -v \
	  --cov=nexus --cov-append --cov-report=term-missing --cov-report=json
	@echo "--- Per-module coverage check ---"
	python scripts/check_coverage.py

# Release: all 8 stages
ci-release: ci-pr
	@echo "--- Stage 6: Benchmarks ---"
	pytest tests/benchmarks/ -v --timeout=120
	@echo "--- Stage 7: Security ---"
	bandit -r nexus/ -ll -q
	safety check --full-report
	@echo "--- Stage 8: Dead Code ---"
	vulture nexus/ --min-confidence 80

# Default `make ci` = PR profile
ci: ci-pr

check: lint test
