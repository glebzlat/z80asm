test:
	python3 -m unittest tests.test_opcodes
	python3 -m unittest tests.test_programs

.PHONY: test
