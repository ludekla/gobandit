AGT := pkg/agent/*
ARM := pkg/arm/*
TRG := main

bin/$(TRG): cmd/$(TRG).go
	go build -o bin/$(TRG) cmd/$(TRG).go

tests:
	go test -v $(AGT)
	go test -v $(ARM)

clean:
	rm bin/*