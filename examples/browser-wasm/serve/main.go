// Command serve hosts the browser-wasm demo over plain HTTP. Camera access
// requires a secure context, which browsers grant to http://localhost.
// The repository's decode benchmark corpus is served under /corpus/ and its
// real-content streams under /realcorpus/ for the decoder page.
//
//	go run ./serve -listen localhost:8090
//
// With -dump DIR, POST /dump/NAME stores the request body as DIR/NAME. The
// encoder page's #verify mode uses it to hand the streams it produced to
// native tools (cmp, dav1d).
package main

import (
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
)

func main() {
	listen := flag.String("listen", "localhost:8090", "HTTP listen address")
	dir := flag.String("dir", "web", "directory to serve")
	corpus := flag.String("corpus", "../../testdata/benchcorpus", "IVF clips served under /corpus/")
	realCorpus := flag.String("realcorpus", "../../testdata/realcorpus", "real-content streams served under /realcorpus/")
	dump := flag.String("dump", "", "directory that receives POST /dump/NAME uploads (disabled when empty)")
	flag.Parse()
	noStore := func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Cache-Control", "no-store")
			h.ServeHTTP(w, r)
		})
	}
	http.Handle("/", noStore(http.FileServer(http.Dir(*dir))))
	http.Handle("/corpus/", noStore(http.StripPrefix("/corpus/", http.FileServer(http.Dir(*corpus)))))
	http.Handle("/realcorpus/", noStore(http.StripPrefix("/realcorpus/", http.FileServer(http.Dir(*realCorpus)))))
	if *dump != "" {
		if err := os.MkdirAll(*dump, 0o755); err != nil {
			log.Fatal(err)
		}
		http.HandleFunc("/dump/", func(w http.ResponseWriter, r *http.Request) {
			name := path.Base(r.URL.Path)
			if r.Method != http.MethodPost || name == "." || name == "/" {
				http.Error(w, "POST /dump/NAME", http.StatusBadRequest)
				return
			}
			body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 256<<20))
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if err := os.WriteFile(filepath.Join(*dump, name), body, 0o644); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			log.Printf("dumped %s (%d bytes)", name, len(body))
		})
	}
	log.Printf("serving %s on http://%s", *dir, *listen)
	log.Fatal(http.ListenAndServe(*listen, nil))
}
