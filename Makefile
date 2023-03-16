AGT := pkg/agent/*
ARM := pkg/arm/*
TRG := soma

bin/$(TRG): cmd/$(TRG).go
	go build -o bin/$(TRG) cmd/$(TRG).go

tests:
	go test -v $(AGT)
	go test -v $(ARM)

clean:
	rm bin/*