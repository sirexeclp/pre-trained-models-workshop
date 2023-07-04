
#!/bin/bash

branch=$(git branch --show-current)

function rebase(){
    git checkout $1
    git rebase $2
}

rebase solutions/backend main
rebase_on_main solutions/benchmark solutions/backend
git checkout $branch