AGT := pkg/agent/*
ARM := pkg/arm/*
TRL := pkg/trial/*
TRG := main

fmt:
	go fmt $(ARM)
	go fmt $(AGT)
	go fmt $(TRL)
	go fmt cmd/*

vet:
	go vet $(ARM)
	go vet $(AGT)
	go vet $(TRL)

test:
	go test -v $(AGT)
	go test -v $(ARM)
	go test -v $(TRL)

build:
	go build -o bin/$(TRG) $(TRG).go

clean:
	rm bin/*
