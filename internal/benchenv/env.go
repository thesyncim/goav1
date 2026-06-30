package benchenv

import (
	"encoding/json"
	"os/exec"
)

var publishBlockedGoEnvVars = []string{
	"GOAMD64",
	"GOARM64",
	"GO386",
	"GOARM",
	"GOMIPS",
	"GOMIPS64",
	"GOPPC64",
	"GOWASM",
	"GOTOOLCHAIN",
	"GOEXPERIMENT",
	"CGO_ENABLED",
	"CC",
	"CXX",
	"GOCACHE",
	"GOMODCACHE",
	"GOPATH",
	"GOTMPDIR",
}

func PublishBlockedGoEnvVars() []string {
	return append([]string(nil), publishBlockedGoEnvVars...)
}

func PublishAmbientGoEnvVars() []string {
	out := []string{"GOFLAGS", "GOMEMLIMIT", "GODEBUG"}
	return append(out, publishBlockedGoEnvVars...)
}

func GoEnvForMetadata() map[string]any {
	out, err := exec.Command("go", "env", "-json").Output()
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	var env map[string]any
	if err := json.Unmarshal(out, &env); err != nil {
		return map[string]any{"error": err.Error()}
	}
	return env
}
