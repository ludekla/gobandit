AGT := pkg/agent/*
ARM := pkg/arm/*
TRG := main

fmt:
	go fmt $(ARM)
	go fmt $(AGT)

test:
	go test -v $(AGT)
	go test -v $(ARM)

build:
	go build -o bin/$(TRG) cmd/$(TRG).go

clean:
	rm bin/*
