# This directory contains load tests for Amazona's API endpoints
# We are using Artillery for performance testing

1- Install artillery
npm i -g artillery@latest

2- run tests
artillery run loadtests/<path>/<test>.yml --output log.json