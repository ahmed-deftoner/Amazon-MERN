# This directory contains load tests for Amazona's API endpoints
# We are using Artillery for performance testing

1- Install artillery
npm i -g artillery@latest

2- run tests
artillery run loadtests/<path>/<test>.yml --output log.json log.html


## Notes
We have used three different phases provided by artillery , warmup phase , ramp up phase , sustained load

For example 
  phases:
    - duration: 60
      arrivalRate: 5
      name: Warm up
    - duration: 120
      arrivalRate: 5
      rampTo: 50
      name: Ramp up load
    - duration: 600
      arrivalRate: 50
      name: Sustained load

The first phase is a slow ramp-up phase to warm up the backend. This phase will send five virtual users to your backend every second for 60 seconds.

The second phase that follows will start with five virtual users and gradually send more users every second for the next two minutes, peaking at 50 virtual users at the end of the phase.

The final phase simulates a sustained spike of 50 virtual users every second for the next ten minutes. This phase is meant to stress test your backend to check the system's sustainability over a more extended period.