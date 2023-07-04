
#!/bin/bash

branch=$(git branch --show-current)

function rebase(){
    git checkout $1
    git rebase $2
}

rebase solutions/benchmark main
rebase solutions/backend solutions/benchmark
git checkout $branch