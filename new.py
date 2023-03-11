from pathlib import Path

txt = '''package main

import (
    "fmt"
)

func main() {
    fmt.Println("Hello Bandit!")
}'''

file = Path('main.go')

if file.exists():
    print("Go file 'main.go' already exists.")
else:
    with file.open(mode='w') as fout:
        fout.write(txt)
