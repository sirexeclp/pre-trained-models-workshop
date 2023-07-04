
#!/bin/bash

branch=$(git branch --show-current)

function rebase(){
    git checkout $1
    git rebase $2
}

rebase solutions/backend main
rebase solutions/benchmark solutions/backend
git checkout $branch