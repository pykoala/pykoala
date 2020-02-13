[![Build Status](https://travis-ci.com/pykoala/koala.svg?branch=master)](https://travis-ci.com/pykoala/koala)

# koala

## Getting started with developing koala

1. Fork koala into your github account
2. Clone your fork onto your laptop:
```
    git clone https://github.com/<your_account>/koala
```
3. Add this repository as another remote (to get the latest stuff):
```
    git remote add upstream https://github.com/pykoala/koala
```
4. Create a branch to work on the changes:
```
    git checkout -b <new_branch>
```
5. Add and commit changes
6. Push up your changes
7. Create a PR, and wait for someone to review it

## Reviewing commits
1. Look through the changes, and provide comments
2. Once the PR is ready, type bors r+, then bors will handle the merge (DON'T
   HIT THE MERGE BUTTON).
3. If the tests fail, help the proposer fix the failures
4. Use bors r+ again

You can also use bors try to try running the tests, without merging
