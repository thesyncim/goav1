package goav1_test

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

// profileClips are the vendored non-superres profile-conformance clips and
// their per-frame aomdec MD5 goldens. They cover profile-1 4:4:4 8/10-bit,
// palette, CDEF/restoration, odd-edge filtering, film grain, profile-2
// 4:2:2/4:2:0, 12-bit film grain, edge motion, 128x128 superblocks, and
// multi-tile streams.
var profileClips = []struct {
	file        string
	frameMD5Hex []string
}{
	{
		file: "profile1-444-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"00211cdc8f799c808849c955a318a0f5",
			"397ff01920ff514bc611ab49d76371c1",
			"f8fbfb25a42da47a7adb71510de9b178",
		},
	},
	{
		file: "profile1-444-8bit-palette-64x64.ivf",
		frameMD5Hex: []string{
			"59e96ccf6f2faf403ae8f8b3214a202f",
			"9a9cfad7d6aead1fb110309d2edc0268",
			"6d7426c5bae2c5cca7c98b584e8057a6",
		},
	},
	{
		file: "profile1-444-10bit-palette-64x64.ivf",
		frameMD5Hex: []string{
			"dcd2ab9753a24747f1b731d011e6478a",
			"b9aaba28bd6cc85fe157befe067cada9",
			"dc714a2fc421619ab9384c185cf21ab6",
		},
	},
	{
		file: "profile1-444-10bit-64x64.ivf",
		frameMD5Hex: []string{
			"96322d4430f0d27243c17e2128cfd625",
			"571c0415f63a1f76e3796360aee3828a",
			"2f6e92f93a95cb4725b9c2f9484ed42e",
		},
	},
	{
		file: "profile1-444-8bit-cdef-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"c7001ffcb04bce20728f246f5494e58a",
			"f1a3ce1e10e2e84e1e61c46e735201ad",
			"1aa1aa327a503776ff58af351198037a",
			"7cfb09c54857fc6dd3994ef09e502fae",
		},
	},
	{
		file: "profile1-444-10bit-cdef-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"851bbe5c81da62f19d13b1efee5b60b5",
			"07da346eb6963d2bf2a7c37eb626b09e",
			"da669f9bf3e0a82727542a1f538b7226",
			"496024c0fd9b1708ca5a9255e317ebf0",
		},
	},
	{
		file: "profile1-444-8bit-edge-cdef-restoration-130x130.ivf",
		frameMD5Hex: []string{
			"8d23e95555d1c482b0c9a9aaf1a82bd0",
			"ccd49de94b63207de8cbc935a5f63e5d",
			"deaa20627918286a339c1c6b30dea119",
		},
	},
	{
		file: "profile1-444-10bit-edge-cdef-restoration-130x130.ivf",
		frameMD5Hex: []string{
			"3fedce7a89a109e6491429ea1b3aa2bc",
			"ff141f5d501503efeed8c32824ec51d4",
			"d996b27d544709b314685f1630599efe",
		},
	},
	{
		file: "profile1-444-8bit-filmgrain-96x96.ivf",
		frameMD5Hex: []string{
			"db1f78af3fbd46eec482cfcc02c59157",
			"f0abb8e97190db604aba5ffae2dd6b47",
			"706e3f8b36da82f717a1496cbbddf0c9",
		},
	},
	{
		file: "profile1-444-10bit-filmgrain-96x96.ivf",
		frameMD5Hex: []string{
			"fc7889878eb34c3fc3044b6353de35a6",
			"d058aaa10ea0893f8bd3d07a52f3ebdc",
			"4e59d8318286bb75836070f420d706a5",
		},
	},
	{
		file: "profile1-444-8bit-inter-64x64.ivf",
		frameMD5Hex: []string{
			"11d7f1cc66c0e0484ad4d7566d66df35",
			"0e590539ec2f2e33419f1de467ab9a18",
			"d829dfffec65fcd80746b20f1c22b21b",
			"2fdb71b5156533455f031e37a96b2a4d",
			"23f8e6ee901994b9cbd2cfb91e181eeb",
			"b69530f8be1ca111af9e9b4c29c3c984",
			"cc93559f8904df46c9a6f3223a3c1f30",
			"fafc19265b25a4380c50ae5a40e4eb27",
		},
	},
	{
		file: "profile1-444-10bit-inter-64x64.ivf",
		frameMD5Hex: []string{
			"3250b3e6c554a9725e372aba0e6f1836",
			"bbcef705eff277b6134c82d154dfd0e0",
			"5588133b22b316a0816a88bda01d711c",
			"fe22846dd23a8ea296132d0b299d772b",
			"d1b019af0ad85b194d474abe0698586b",
			"9d6cf7fd0eda963a93d553b7716e76b4",
			"5fb4e9e5ccee31c503b160de7a173c03",
			"ff7ee0fb4599801fc3015b9dd744ab1f",
		},
	},
	{
		file: "profile1-444-8bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"45b7de404cb0b5d469958768b8f5d479",
			"945ed5c3b0a1191eed1499b6f3425025",
			"35f885d3ec1b37fb61f896b78f4a7ec8",
			"e852ba422c8e6c6e0d4e9077c1751f8d",
			"956fbda68f31ead438b5835aa5f87980",
		},
	},
	{
		file: "profile1-444-10bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"56de8e3645276aa7322d4d26ddf4d048",
			"490eb309b57743705ad4a941b5d8dc89",
			"da2ff78bcd92690494532709c4d1e6a9",
			"f31fb72982c6141082dc7864e341f47a",
			"c03c86fb63b5a556b37f729b344a2efe",
		},
	},
	{
		file: "profile1-444-10bit-multitile-inter-2x1-128x128.ivf",
		frameMD5Hex: []string{
			"622e6159c052250bebf439858a73ec97",
			"5c3fd1c4289cb96721ec8ab01b26f039",
			"e60e744ca2d50f4701952b15fda80cfe",
			"cc6ee6ed27717c5341606e3d4371affb",
			"e31c6c9af577991ed02f043a0523ada8",
		},
	},
	{
		file: "profile2-422-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"dabd492413632a810adeaf4e5d0c6d97",
			"eb6cf8d1d4d644686cf03c513acd5978",
			"84983e98afeef6692448e98fc5980431",
		},
	},
	{
		file: "profile2-422-10bit-66x66.ivf",
		frameMD5Hex: []string{
			"8f6b2f543fdd300f6a09285bd35b2dea",
			"e6137fd48ed1c842cfef45b6a13d09e9",
			"d66f267df360523a90ae1f25a101f68c",
		},
	},
	{
		file: "profile2-422-8bit-inter-64x64.ivf",
		frameMD5Hex: []string{
			"e10a47c64fa7318c45ce0762bea3f0ce",
			"393c471479eff858fe4901cee94b6e57",
			"6251e9f45b4733552ba2b687b4c38154",
			"43fbaf637130f61feaaed7e8844c5d4b",
			"7ae17990bf39d7b507a441095ebe1ecb",
			"275861a447790a60e26cf3f574cab809",
			"61013488539b1b1a169f1e372d96d180",
			"0eca225d9dd9f6f768ab49a216511304",
		},
	},
	{
		file: "profile2-420-12bit-64x64.ivf",
		frameMD5Hex: []string{
			"e714741e4ad4fce5a4469c79705f132c",
			"447103c7d7358e4cbb6f5b98ce4e1be1",
			"dcd86cbbf80f81d9699eaf6c6e879e72",
		},
	},
	{
		file: "profile2-420-12bit-66x66.ivf",
		frameMD5Hex: []string{
			"f8dc71c2aafcf0ed77ce9d91a0a8ca0d",
			"8d44478c32115672846d4579d3962b58",
			"9825261dad9f100b51cba05ad534029d",
		},
	},
	{
		file: "profile2-420-12bit-filmgrain-96x96.ivf",
		frameMD5Hex: []string{
			"48d28098e31d5ef96fa13d69bf13a374",
			"774b0173bb05d4d3d271d3e04303ac1b",
			"490cb07ef4b11d39480b11947796ef4b",
		},
	},
	{
		file: "superblock128-420-8bit-128x128.ivf",
		frameMD5Hex: []string{
			"42ace8044951a222256ffcd65aa8bb67",
			"c1b26993930b06dec4f0d05457035565",
			"4e7b4292d96e53cc8ea5b16c8d7e13de",
		},
	},
	{
		file: "profile0-420-8bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"f6c673ec13ccdd994fd10373a1d7f814",
			"a3b01764f4b3e9c7bcec7c0e7e86be64",
			"8158c19b78deb9ac6aaab9d411fe268f",
			"eebdcf491352f1b13a24b614ba137832",
			"54582e43045ece242c0a50e3f73e7a6b",
		},
	},
	{
		file: "profile2-420-12bit-edgemv-130x130.ivf",
		frameMD5Hex: []string{
			"f85baff9f8f516879ab8da538e582a09",
			"5dfd30b2a7eb7e2178a06bd7e060bc80",
			"f14ff5bc1a1504908f1d22b4be5a0ca3",
			"24e96db2d3644a748e587dd011863a4a",
			"325a954def1c43d852c980ddc66ba652",
		},
	},
	{
		file: "profile1-444-8bit-multitile-2x1-256x256.ivf",
		frameMD5Hex: []string{
			"afb8da499aa3e0394b61ae7859323bd7",
			"eaf99aece59fc4ecbfb556eee5fcd3fc",
			"7433db9f86a39411f616573b5735aad5",
			"3661ba977cd76893d39563346fb3f5ec",
		},
	},
	{
		file: "multitile-2x1-rows-256x256.ivf",
		frameMD5Hex: []string{
			"19abff965e7c0b982b779b14ce5d67a9",
			"d9670540a9ef1cc444eb684dde65928c",
			"1ece7ab7594b417ad313c395c9bf153c",
			"6de64107acac0d89ad361dfe615a1c5f",
		},
	},
}

func TestSimpleDecoderZeroValueGuards(t *testing.T) {
	var zero av1.Decoder
	if frames, ok, err := zero.DecodeNext(); err == nil || ok || frames != nil {
		t.Fatalf("zero DecodeNext frames=%v ok=%v err=%v", frames, ok, err)
	}
	if frames, err := zero.DecodeAll(); err == nil || frames != nil {
		t.Fatalf("zero DecodeAll frames=%v err=%v", frames, err)
	}
	if err := zero.Reset(); err == nil {
		t.Fatal("zero Reset returned nil error")
	}
	zero.Close()

	var nilDecoder *av1.Decoder
	if frames, ok, err := nilDecoder.DecodeNext(); err == nil || ok || frames != nil {
		t.Fatalf("nil DecodeNext frames=%v ok=%v err=%v", frames, ok, err)
	}
	if frames, err := nilDecoder.DecodeAll(); err == nil || frames != nil {
		t.Fatalf("nil DecodeAll frames=%v err=%v", frames, err)
	}
	if err := nilDecoder.Reset(); err == nil {
		t.Fatal("nil Reset returned nil error")
	}
	nilDecoder.Close()
}

var superResProfileClips = []struct {
	file        string
	frameMD5Hex []string
}{
	{
		file: "profile1-444-8bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"315663b1b307e46d4d4719ad759b39c2",
			"b1c26bbbf277ac82c7fbdc0807974a1f",
			"bd6aad53bf088c7ff5f54d93c59ff649",
			"775085d7d08c8608b0acd2defa0fdc58",
		},
	},
	{
		file: "profile1-444-8bit-superres-inter-160x128.ivf",
		frameMD5Hex: []string{
			"7057484f2d2048053692a9bc41ce197b",
			"6b136fd484722654825fe6abc2e1773a",
			"cddb59775cd003ed4624b782fce4eca4",
			"32e3240e78dd8f53e7b673c9120fbe86",
			"5054adcd48bad5490911bb07248fea75",
			"150acaa0e5a682d454d7538b15cc4f2f",
			"5d0ab2de3f3e9ce25036226b11375e49",
			"78c4f82e984b44a272930333264bc764",
		},
	},
	{
		file: "profile1-444-10bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"c556c65e1ceceb72778a36a5351b1d96",
			"6e66c5c97438754259295a95dde348a4",
			"cbf0f9d0584eaaf09538091edfb8f885",
			"f6f4c4213052d92f0ae700e793cad629",
		},
	},
	{
		file: "profile1-444-10bit-superres-restoration-160x128.ivf",
		frameMD5Hex: []string{
			"7946827a2d34b524ed92cc9a3b8504b8",
			"5c9585eecc0bff58859f42933505bdef",
			"92508e59e357059ea42166a6befead4f",
			"883da41a575466a10360f4b90710a339",
		},
	},
	{
		file: "profile2-420-12bit-superres-160x128.ivf",
		frameMD5Hex: []string{
			"ea09f096227bac3e03277487354337fc",
			"2f144bf1f5144412cb606e79f8d74770",
			"92beb6d99acf1d73926dbed7b1eeb1ea",
			"17ae4f314b20aecc1d20a83bdee07da0",
		},
	},
	{
		file: "profile1-444-10bit-superres-inter-static-160x128.ivf",
		frameMD5Hex: []string{
			"ab4284f9b59b7cfd81bdf5ab27d7e10b",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
			"7a72c06659a4a041bbdcf65a7f45bb83",
		},
	},
	{
		file: "profile1-444-10bit-superres-inter-simple-160x128.ivf",
		frameMD5Hex: []string{
			"0a0b14f62deee8bdcedbdd2c648c6396",
			"e7d1a6243d38d554df4f63b2a9d1beaf",
			"51bb916a676b953252f33e514dab46fd",
			"15a0daafab712254269a4ce27800c1c9",
			"8114f35db56233a52e75e367eb9f7a33",
			"b5dfcdadffe2c9d246ad9930e39ab290",
			"36278c9c6a7e15d325a45887e11ab60a",
			"8e42428b236d37ca8d59c5c5e4ece89e",
		},
	},
	{
		file: "superres-420-8bit-160x128.ivf",
		frameMD5Hex: []string{
			"024e5d1f55340eee39893567e8000554",
			"80c239661f2cd1afe5fce5038ceac76f",
			"93dc5cde59cb99bd524264b15c70a700",
			"87887afdc06ba5f0e88b2acbe1ce0ce5",
		},
	},
	{
		file: "superres-restoration-420-8bit-160x128.ivf",
		frameMD5Hex: []string{
			"93533110e2bd60fed6f15b5e65178bc6",
			"bf83a513a868584aec7f9f05c5187d70",
			"4e7dc4a0d7ab9cebdbdfdf3ef73b0d1c",
			"6af9472b0679093facbefa85e973a7f9",
		},
	},
}

func profileClipPath(name string) string {
	return filepath.Join("internal", "av1", "testvector", "testdata", "profiles", name)
}

func TestSimpleDecoderProfileClipTablesCoverVendoredProfiles(t *testing.T) {
	want := vendoredProfileClipNames(t)
	got := make(map[string]int, len(profileClips)+len(superResProfileClips))
	for _, clip := range profileClips {
		got[clip.file]++
	}
	for _, clip := range superResProfileClips {
		got[clip.file]++
	}

	for file, count := range got {
		if count != 1 {
			t.Errorf("profile clip %s appears %d times in public decoder tables", file, count)
		}
		if _, ok := want[file]; !ok {
			t.Errorf("public decoder table references missing vendored clip %s", file)
		}
	}
	for file := range want {
		if got[file] == 0 {
			t.Errorf("vendored profile clip %s is not covered by public decoder tables", file)
		}
	}
}

func vendoredProfileClipNames(t *testing.T) map[string]struct{} {
	t.Helper()
	paths, err := filepath.Glob(profileClipPath("*.ivf"))
	if err != nil {
		t.Fatalf("glob profile clips: %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("no vendored profile clips found")
	}
	clips := make(map[string]struct{}, len(paths))
	for _, path := range paths {
		clips[filepath.Base(path)] = struct{}{}
	}
	return clips
}

// TestSimpleDecoderProfileClipsMatchGolden decodes each vendored profile clip
// through the high-level Decoder API and asserts every visible frame's MD5
// equals the libaom golden. This proves the convenience API drives the same
// byte-exact public decode path as the low-level hand-bound sequence.
func TestSimpleDecoderProfileClipsMatchGolden(t *testing.T) {
	for _, clip := range profileClips {
		t.Run(clip.file, func(t *testing.T) {
			ivf, err := os.ReadFile(profileClipPath(clip.file))
			if err != nil {
				t.Fatalf("read clip: %v", err)
			}

			dec, err := av1.NewDecoderFromIVF(ivf)
			if err != nil {
				t.Fatalf("NewDecoderFromIVF: %v", err)
			}
			defer dec.Close()

			var got []string
			for {
				frames, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatalf("DecodeNext: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range frames {
					sum := frameMD5Visible(f)
					got = append(got, hex.EncodeToString(sum[:]))
				}
			}

			if len(got) != len(clip.frameMD5Hex) {
				t.Fatalf("decoded %d visible frames, want %d", len(got), len(clip.frameMD5Hex))
			}
			for i, want := range clip.frameMD5Hex {
				if got[i] != want {
					t.Fatalf("frame %d md5 got=%s want=%s (libaom golden)", i, got[i], want)
				}
			}
		})
	}
}

// TestSimpleDecoderSuperResProfileClipsMatchGolden proves the high-level
// Decoder automatically routes super-res streams through the external reference
// publication path, so inter frames reference the upscaled output surface rather
// than the coded reconstruction surface.
func TestSimpleDecoderSuperResProfileClipsMatchGolden(t *testing.T) {
	for _, clip := range superResProfileClips {
		t.Run(clip.file, func(t *testing.T) {
			ivf, err := os.ReadFile(profileClipPath(clip.file))
			if err != nil {
				t.Fatalf("read clip: %v", err)
			}

			dec, err := av1.NewDecoderFromIVF(ivf)
			if err != nil {
				t.Fatalf("NewDecoderFromIVF: %v", err)
			}
			defer dec.Close()

			var got []string
			for {
				frames, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatalf("DecodeNext: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range frames {
					sum := frameMD5Visible(f)
					got = append(got, hex.EncodeToString(sum[:]))
				}
			}

			if len(got) != len(clip.frameMD5Hex) {
				t.Fatalf("decoded %d visible frames, want %d", len(got), len(clip.frameMD5Hex))
			}
			for i, want := range clip.frameMD5Hex {
				if got[i] != want {
					t.Fatalf("frame %d md5 got=%s want=%s (libaom golden)", i, got[i], want)
				}
			}
		})
	}
}

func TestNewDecoderFromIVFReaderAtSuperResMatchesGolden(t *testing.T) {
	clip := superResProfileClips[1]
	ivf, err := os.ReadFile(profileClipPath(clip.file))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}

	dec, header, err := av1.NewDecoderFromIVFReaderAt(bytes.NewReader(ivf), int64(len(ivf)))
	if err != nil {
		t.Fatalf("NewDecoderFromIVFReaderAt: %v", err)
	}
	defer dec.Close()
	if header.FrameCount != uint32(len(clip.frameMD5Hex)) {
		t.Fatalf("reader header frame count=%d want %d", header.FrameCount, len(clip.frameMD5Hex))
	}

	var got []string
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range frames {
			sum := frameMD5Visible(f)
			got = append(got, hex.EncodeToString(sum[:]))
		}
	}
	if len(got) != len(clip.frameMD5Hex) {
		t.Fatalf("decoded %d visible frames, want %d", len(got), len(clip.frameMD5Hex))
	}
	for i, want := range clip.frameMD5Hex {
		if got[i] != want {
			t.Fatalf("frame %d md5 got=%s want=%s (libaom golden)", i, got[i], want)
		}
	}
}

// TestDecodeIVFMatchesGolden exercises the one-shot DecodeIVF helper and asserts
// each returned independent frame copy hashes to the libaom golden.
func TestDecodeIVFMatchesGolden(t *testing.T) {
	clip := profileClips[0]
	ivf, err := os.ReadFile(profileClipPath(clip.file))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}

	frames, err := av1.DecodeIVF(ivf)
	if err != nil {
		t.Fatalf("DecodeIVF: %v", err)
	}
	if len(frames) != len(clip.frameMD5Hex) {
		t.Fatalf("decoded %d frames, want %d", len(frames), len(clip.frameMD5Hex))
	}
	for i, want := range clip.frameMD5Hex {
		sum := decodedFrameMD5(frames[i])
		if g := hex.EncodeToString(sum[:]); g != want {
			t.Fatalf("frame %d md5 got=%s want=%s", i, g, want)
		}
	}
}

func TestSimpleDecoderRejectsMalformedIVFInputs(t *testing.T) {
	lowOverhead := func(raw ...byte) []byte {
		return append([]byte(nil), raw...)
	}
	sizedOBU := func(header byte, size uint32, payload ...byte) []byte {
		out := []byte{header}
		out = appendPublicLEB128(out, size)
		out = append(out, payload...)
		return out
	}
	truncatedIVF := appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{{
		payload: []byte{0xaa, 0xbb},
	}})
	truncatedIVF = truncatedIVF[:len(truncatedIVF)-1]
	shortFrameHeaderIVF := appendPublicIVF(nil, 16, 16, 30, 1, nil)
	shortFrameHeaderIVF = append(shortFrameHeaderIVF, 0x01)

	tests := []struct {
		name    string
		ivf     []byte
		wantErr error
	}{
		{
			name:    "short IVF frame header",
			ivf:     shortFrameHeaderIVF,
			wantErr: av1.ErrIVFShortFrameHeader,
		},
		{
			name:    "short IVF frame payload",
			ivf:     truncatedIVF,
			wantErr: av1.ErrIVFShortFramePayload,
		},
		{
			name: "low overhead OBU missing size field",
			ivf: appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{{
				payload: lowOverhead(0x10),
			}}),
			wantErr: av1.ErrOBUMissingSizeField,
		},
		{
			name: "low overhead OBU short payload",
			ivf: appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{{
				payload: sizedOBU(0x12, 4, 0xaa),
			}}),
			wantErr: av1.ErrOBUShortPayload,
		},
		{
			name: "forbidden OBU header bit",
			ivf: appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{{
				payload: lowOverhead(0x80),
			}}),
			wantErr: av1.ErrOBUForbiddenBit,
		},
		{
			name: "reserved OBU header bit",
			ivf: appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{{
				payload: lowOverhead(0x13, 0x00),
			}}),
			wantErr: av1.ErrOBUReservedBit,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name+"/NewDecoderFromIVF", func(t *testing.T) {
			dec, err := av1.NewDecoderFromIVF(tc.ivf)
			if dec != nil {
				dec.Close()
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("NewDecoderFromIVF err=%v want %v", err, tc.wantErr)
			}
		})
		t.Run(tc.name+"/NewDecoderFromIVFReaderAt", func(t *testing.T) {
			dec, _, err := av1.NewDecoderFromIVFReaderAt(bytes.NewReader(tc.ivf), int64(len(tc.ivf)))
			if dec != nil {
				dec.Close()
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("NewDecoderFromIVFReaderAt err=%v want %v", err, tc.wantErr)
			}
		})
		t.Run(tc.name+"/DecodeIVF", func(t *testing.T) {
			_, err := av1.DecodeIVF(tc.ivf)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("DecodeIVF err=%v want %v", err, tc.wantErr)
			}
		})
	}
}

func TestSimpleDecoderRejectsSwitchFrameWithoutReferences(t *testing.T) {
	payload := appendPublicLowOverheadOBU(nil, av1.OBUSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	switchFrame := append([]byte{}, publicSimpleDecoderSwitchFrameHeaderPayload()...)
	switchFrame = append(switchFrame, 0xbb)
	payload = appendPublicLowOverheadOBU(payload, av1.OBUFrame, switchFrame)
	ivf := appendPublicIVF(nil, 16, 9, 30, 1, []publicIVFFrame{{payload: payload}})

	t.Run("NewDecoderFromIVF", func(t *testing.T) {
		dec, err := av1.NewDecoderFromIVF(ivf)
		if dec != nil {
			dec.Close()
		}
		if !errors.Is(err, av1.ErrReferenceFrameNeeded) {
			t.Fatalf("NewDecoderFromIVF err=%v want %v", err, av1.ErrReferenceFrameNeeded)
		}
	})

	t.Run("DecodeIVF", func(t *testing.T) {
		_, err := av1.DecodeIVF(ivf)
		if !errors.Is(err, av1.ErrReferenceFrameNeeded) {
			t.Fatalf("DecodeIVF err=%v want %v", err, av1.ErrReferenceFrameNeeded)
		}
	})
}

func TestSimpleDecoderTileListIVFErrors(t *testing.T) {
	validPayload := av1.AppendTileListOBU(nil, av1.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []av1.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      0,
			AnchorTileCol:      0,
			TileDataSizeMinus1: 2,
			TileData:           []byte{0xaa, 0xbb, 0xcc},
		}},
	})

	tests := []struct {
		name        string
		tilePayload []byte
		wantErr     error
	}{
		{
			name:        "valid tile list is unsupported playback",
			tilePayload: validPayload,
			wantErr:     av1.ErrDecoderUnsupportedTileList,
		},
		{
			name:        "malformed tile list propagates parse error",
			tilePayload: []byte{0x00, 0x00, 0x00, 0x00},
			wantErr:     av1.ErrTileListShortEntry,
		},
		{
			name:        "invalid tile count propagates parse error",
			tilePayload: []byte{0xff, 0xff, 0xff, 0xff},
			wantErr:     av1.ErrTileListInvalidTileCount,
		},
		{
			name:        "short tile data propagates parse error",
			tilePayload: []byte{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0x00, 0x04, 0xaa},
			wantErr:     av1.ErrTileListShortTileData,
		},
		{
			name:        "trailing tile-list bytes propagate parse error",
			tilePayload: append(append([]byte{}, validPayload...), 0xff),
			wantErr:     av1.ErrTileListTrailingBytes,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prime := firstProfilePayload(t)
			payload := appendPublicLowOverheadOBU(nil, av1.OBUTileList, tc.tilePayload)
			ivf := appendPublicIVF(nil, 64, 64, 30, 1, []publicIVFFrame{
				{payload: prime},
				{timestamp: 1, payload: payload},
			})

			dec, err := av1.NewDecoderFromIVF(ivf)
			if err != nil {
				t.Fatalf("NewDecoderFromIVF: %v", err)
			}
			defer dec.Close()

			frames, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("DecodeNext prime: %v", err)
			}
			if !ok || len(frames) == 0 {
				t.Fatalf("DecodeNext prime frames=%d ok=%v", len(frames), ok)
			}

			frames, ok, err = dec.DecodeNext()
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("DecodeNext err=%v want %v", err, tc.wantErr)
			}
			if ok || len(frames) != 0 {
				t.Fatalf("DecodeNext frames=%d ok=%v after tile-list error", len(frames), ok)
			}
		})
	}
}

func publicSimpleDecoderSwitchFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false) // show_existing_frame
	w.writeBits(uint64(av1.FrameTypeSwitch), 2)
	w.writeBool(true)  // show_frame
	w.writeBool(false) // disable_cdf_update
	for range av1.InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBits(15, 4) // frame_width_minus_1
	w.writeBits(8, 4)  // frame_height_minus_1
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(0, 2)  // interpolation_filter = EIGHTTAP
	w.writeBool(false) // is_motion_mode_switchable
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	publicDecoderResidualWriteColorQuantParams(&w)
	publicDecoderResidualWriteZeroSegmentationParams(&w)
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for range av1.InterRefsPerFrame {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func firstProfilePayload(t *testing.T) []byte {
	t.Helper()

	ivf, err := os.ReadFile(profileClipPath(profileClips[0].file))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}
	it, err := av1.NewIVFIterator(ivf)
	if err != nil {
		t.Fatalf("NewIVFIterator: %v", err)
	}
	frame, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first profile frame ok=%v err=%v", ok, err)
	}
	return append([]byte(nil), frame.Payload...)
}

func TestDecodeIVFSuperResProfileClipsMatchGolden(t *testing.T) {
	for _, clip := range superResProfileClips {
		t.Run(clip.file, func(t *testing.T) {
			ivf, err := os.ReadFile(profileClipPath(clip.file))
			if err != nil {
				t.Fatalf("read clip: %v", err)
			}

			frames, err := av1.DecodeIVF(ivf)
			if err != nil {
				t.Fatalf("DecodeIVF: %v", err)
			}
			if len(frames) != len(clip.frameMD5Hex) {
				t.Fatalf("decoded %d frames, want %d", len(frames), len(clip.frameMD5Hex))
			}
			for i, want := range clip.frameMD5Hex {
				sum := decodedFrameMD5(frames[i])
				if g := hex.EncodeToString(sum[:]); g != want {
					t.Fatalf("frame %d md5 got=%s want=%s", i, g, want)
				}
			}
		})
	}
}

func TestSimpleDecoderPostFilteredClipAllocBudget(t *testing.T) {
	tests := []struct {
		name        string
		file        string
		wantVisible int
		maxAllocs   float64
	}{
		{
			name:        "postfiltered cdef restoration",
			file:        "profile1-444-8bit-cdef-restoration-160x128.ivf",
			wantVisible: 4,
			maxAllocs:   0,
		},
		{
			name:        "superres inter external references",
			file:        "profile1-444-8bit-superres-inter-160x128.ivf",
			wantVisible: 8,
			maxAllocs:   0,
		},
		{
			name:        "superres restoration high bit depth",
			file:        "profile1-444-10bit-superres-restoration-160x128.ivf",
			wantVisible: 4,
			maxAllocs:   0,
		},
		{
			name:        "superres inter high bit depth",
			file:        "profile1-444-10bit-superres-inter-simple-160x128.ivf",
			wantVisible: 8,
			maxAllocs:   0,
		},
		{
			name:        "film grain high bit depth",
			file:        "profile1-444-10bit-filmgrain-96x96.ivf",
			wantVisible: 3,
			maxAllocs:   0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assertSimpleDecoderAllocBudget(t, tc.file, tc.wantVisible, tc.maxAllocs)
		})
	}
}

func TestSimpleDecoderSuperResColdAndWarmAllocBudget(t *testing.T) {
	tests := []struct {
		name        string
		file        string
		wantVisible int
	}{
		{
			name:        "superres inter external references",
			file:        "profile1-444-8bit-superres-inter-160x128.ivf",
			wantVisible: 8,
		},
		{
			name:        "superres restoration high bit depth",
			file:        "profile1-444-10bit-superres-restoration-160x128.ivf",
			wantVisible: 4,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assertSimpleDecoderColdAndWarmAllocBudget(t, tc.file, tc.wantVisible)
		})
	}
}

func assertSimpleDecoderAllocBudget(t *testing.T, file string, wantVisible int, maxAllocs float64) {
	t.Helper()

	ivf, err := os.ReadFile(profileClipPath(file))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}

	dec, err := av1.NewDecoderFromIVF(ivf)
	if err != nil {
		t.Fatalf("NewDecoderFromIVF: %v", err)
	}
	defer dec.Close()

	decodeAll := func() {
		if err := dec.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		visible := 0
		for {
			frames, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("DecodeNext: %v", err)
			}
			if !ok {
				break
			}
			visible += len(frames)
		}
		if visible != wantVisible {
			t.Fatalf("decoded %d visible frames, want %d", visible, wantVisible)
		}
	}

	decodeAll()
	if runtime.GOOS == "windows" && (file == "profile1-444-8bit-superres-inter-160x128.ivf" || file == "profile1-444-10bit-superres-inter-simple-160x128.ivf") {
		t.Skip("windows runtime allocation accounting is not stable for the superres external-reference path")
	}
	allocs := testing.AllocsPerRun(20, decodeAll)
	if allocs > maxAllocs {
		t.Fatalf("%s DecodeNext allocs/run=%f want <= %f", file, allocs, maxAllocs)
	}
}

func assertSimpleDecoderColdAndWarmAllocBudget(t *testing.T, file string, wantVisible int) {
	t.Helper()

	if runtime.GOOS == "windows" && file == "profile1-444-8bit-superres-inter-160x128.ivf" {
		t.Skip("windows runtime allocation accounting is not stable for the superres external-reference path")
	}

	ivf, err := os.ReadFile(profileClipPath(file))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}

	const runs = 20
	decs := make([]*av1.Decoder, runs+1)
	for i := range decs {
		dec, err := av1.NewDecoderFromIVF(ivf)
		if err != nil {
			t.Fatalf("NewDecoderFromIVF[%d]: %v", i, err)
		}
		decs[i] = dec
		defer dec.Close()
	}
	next := 0
	coldAllocs := testing.AllocsPerRun(runs, func() {
		if next >= len(decs) {
			t.Fatalf("cold decoder index %d out of %d", next, len(decs))
		}
		assertSimpleDecoderDecodeAllVisible(t, decs[next], wantVisible)
		next++
	})
	if coldAllocs != 0 {
		t.Fatalf("%s cold DecodeNext allocs/run=%f want 0", file, coldAllocs)
	}

	dec, err := av1.NewDecoderFromIVF(ivf)
	if err != nil {
		t.Fatalf("NewDecoderFromIVF warm: %v", err)
	}
	defer dec.Close()
	assertSimpleDecoderDecodeAllVisible(t, dec, wantVisible)
	warmAllocs := testing.AllocsPerRun(runs, func() {
		if err := dec.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		assertSimpleDecoderDecodeAllVisible(t, dec, wantVisible)
	})
	if warmAllocs != 0 {
		t.Fatalf("%s warm DecodeNext allocs/run=%f want 0", file, warmAllocs)
	}
}

func assertSimpleDecoderDecodeAllVisible(t *testing.T, dec *av1.Decoder, wantVisible int) {
	t.Helper()
	visible := 0
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		visible += len(frames)
	}
	if visible != wantVisible {
		t.Fatalf("decoded %d visible frames, want %d", visible, wantVisible)
	}
}

// TestSimpleDecoderMatchesLowLevelPath proves byte-for-byte that the high-level
// Decoder produces exactly the same visible-frame bytes as the low-level
// hand-bound stream-runner path for every vendored profile clip.
func TestSimpleDecoderMatchesLowLevelPath(t *testing.T) {
	for _, clip := range profileClips {
		t.Run(clip.file, func(t *testing.T) {
			ivf, err := os.ReadFile(profileClipPath(clip.file))
			if err != nil {
				t.Fatalf("read clip: %v", err)
			}

			lowLevel := decodeLowLevel(t, ivf)

			dec, err := av1.NewDecoderFromIVF(ivf)
			if err != nil {
				t.Fatalf("NewDecoderFromIVF: %v", err)
			}
			defer dec.Close()

			var highLevel [][16]byte
			for {
				frames, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatalf("DecodeNext: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range frames {
					highLevel = append(highLevel, frameMD5Visible(f))
				}
			}

			if len(highLevel) != len(lowLevel) {
				t.Fatalf("high-level produced %d frames, low-level %d", len(highLevel), len(lowLevel))
			}
			for i := range lowLevel {
				if highLevel[i] != lowLevel[i] {
					t.Fatalf("frame %d differs: high=%s low=%s",
						i, hex.EncodeToString(highLevel[i][:]), hex.EncodeToString(lowLevel[i][:]))
				}
			}
		})
	}
}

func TestPublicDecoderRTPPayloadRunnerMatchesLowOverhead(t *testing.T) {
	const width, height = 192, 128
	enc, err := av1.NewRTCEncoderWithConfig(av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: width, Height: height},
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    120,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 420,
		Scalability:       av1.EncoderScalabilityModeL1T2,
	})
	if err != nil {
		t.Fatalf("NewRTCEncoderWithConfig: %v", err)
	}

	var lowOverheads [][]byte
	var rtpPayloads [][]byte
	for i := 0; i < 4; i++ {
		frame, err := enc.Encode(publicRTCMatrixFrame(width, height, i), false)
		if err != nil {
			t.Fatalf("Encode frame %d: %v", i, err)
		}
		lowOverheads = append(lowOverheads, append([]byte(nil), frame.Data...))
		rtpPayloads = append(rtpPayloads, publicDecoderRTPPayloadsForFrame(t, frame)...)
	}

	want := decodeLowOverheadPayloads(t, lowOverheads)
	got := decodeRTPPayloadsLowLevel(t, rtpPayloads)
	if len(got) != len(want) {
		t.Fatalf("RTP decoded %d frames, low-overhead decoded %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frame %d digest differs: rtp=%s low=%s",
				i, hex.EncodeToString(got[i][:]), hex.EncodeToString(want[i][:]))
		}
	}
}

func publicDecoderRTPPayloadsForFrame(t *testing.T, frame av1.RTCFrame) [][]byte {
	t.Helper()
	limits := av1.RTPPayloadSizeLimits{MaxPayloadLen: 96}
	firstSize, err := frame.RTPPacketScratchLen(limits, nil)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen first: %v", err)
	}
	obuScratch := make([]av1.RTPPacketizerOBU, firstSize.Packetizer.OBUs)
	size, err := frame.RTPPacketScratchLen(limits, obuScratch)
	if err != nil {
		t.Fatalf("RTPPacketScratchLen full: %v", err)
	}
	packetScratch := make([]av1.RTPPacketPlan, size.Packetizer.Packets)
	workScratch := make([]av1.RTPPacketPlan, size.Packetizer.Work)
	payloadBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxPayloadBytes)
	descriptorBuf := make([]byte, 0, size.Packetizer.Packets*size.MaxDescriptorBytes)
	spans := make([]av1.EncoderWebRTCRTPPacketSpan, size.Packetizer.Packets)
	rtpPayloads, _, packetCount, err := frame.AppendRTPPackets(payloadBuf, descriptorBuf, spans, limits, obuScratch, packetScratch, workScratch)
	if err != nil {
		t.Fatalf("AppendRTPPackets: %v", err)
	}
	out := make([][]byte, packetCount)
	for i := 0; i < packetCount; i++ {
		span := spans[i]
		out[i] = append([]byte(nil), rtpPayloads[span.PayloadOffset:span.PayloadOffset+span.PayloadLength]...)
	}
	return out
}

// decodeLowLevel reproduces the hand-bound public stream-runner path (as in
// cmd/aom-go-dec and conformance/publicpath_test.go) and returns the per-frame
// visible MD5 in display order, copying pixels out per payload so multi-frame
// streams retain every frame.
func decodeLowLevel(t *testing.T, ivf []byte) [][16]byte {
	t.Helper()

	it, err := av1.NewIVFIterator(ivf)
	if err != nil {
		t.Fatalf("NewIVFIterator: %v", err)
	}
	var payloads [][]byte
	for {
		f, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf next: %v", err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, append([]byte(nil), f.Payload...))
	}

	const workers = 1
	workerPool, err := av1.NewTileWorkerPool(workers)
	if err != nil {
		t.Fatalf("worker pool: %v", err)
	}
	defer workerPool.Close()

	var probeStream av1.DecoderStream
	probeEvents := make([]av1.DecoderEvent, 16*len(payloads)+64)
	probeSpans := make([]av1.TileSpan, av1.MaxTiles)
	probeJobs := make([]av1.TileJob, av1.MaxTiles)
	probeBatches := make([]av1.TileBatch, av1.MaxTiles)
	plan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(
		probeStream, payloads, workers, probeEvents, probeSpans, probeJobs, probeBatches)
	if err != nil {
		t.Fatalf("stream plan: %v", err)
	}
	if !plan.HasEvent() {
		t.Fatal("stream plan did not identify a bind event")
	}

	format, err := av1.FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
	}

	const surfaceCount = av1.RefFrames + 1
	_, backingSize, err := av1.FramePoolRequiredSize(format, surfaceCount)
	if err != nil {
		t.Fatalf("FramePoolRequiredSize: %v", err)
	}
	pool, err := av1.BindFramePool(make([]byte, backingSize), format,
		make([]av1.Frame, surfaceCount), make([]int, surfaceCount), make([]bool, surfaceCount))
	if err != nil {
		t.Fatalf("BindFramePool: %v", err)
	}

	var (
		stream     av1.DecoderStream
		refs       av1.DecoderSurfaceReferences
		state      av1.DecoderFrameWorkState
		stats      av1.DecoderFrameWorkTileResidualStats
		sideData   av1.DecoderFrameWorkSideData
		batch      av1.DecoderFrameWorkBatchResidualRunner
		postFilter av1.DecoderFrameWorkReusableSupportedPostFilterRunner
	)
	scratch := lowLevelStreamScratch(plan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream,
		av1.DecoderFrameWorkResidualEventRuntime{
			State:             &state,
			Refs:              &refs,
			FramePool:         &pool,
			Align:             64,
			ReferenceSurfaces: make([]int, av1.InterRefsPerFrame),
			ReferenceFrames:   make([]*av1.Frame, av1.InterRefsPerFrame),
			Releases:          make([]int, av1.RefFrames),
			WorkerPool:        workerPool,
			SideData:          &sideData,
			Stats:             &stats,
		}, scratch, &batch)
	if err != nil {
		t.Fatalf("bind runner: %v", err)
	}

	var digests [][16]byte
	for i, payload := range payloads {
		var result av1.DecoderFrameWorkResidualStreamResult
		if err := runner.RunLowOverheadIntoWithPostFilterRunner(&result, payload, &postFilter); err != nil {
			t.Fatalf("frame %d run: %v", i, err)
		}
		for _, f := range result.Run.Outputs {
			if f != nil {
				digests = append(digests, frameMD5Visible(f))
			}
		}
	}
	return digests
}

func decodeLowOverheadPayloads(t *testing.T, payloads [][]byte) [][16]byte {
	t.Helper()
	dec, err := av1.NewDecoder(payloads)
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	defer dec.Close()
	var digests [][16]byte
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range frames {
			digests = append(digests, frameMD5Visible(f))
		}
	}
	return digests
}

func decodeRTPPayloadsLowLevel(t *testing.T, payloads [][]byte) [][16]byte {
	t.Helper()
	const workers = 1
	workerPool, err := av1.NewTileWorkerPool(workers)
	if err != nil {
		t.Fatalf("worker pool: %v", err)
	}
	defer workerPool.Close()

	rtpBufferBytes := 0
	for _, payload := range payloads {
		rtpBufferBytes += len(payload)
	}
	var probeStream av1.DecoderStream
	probeEvents := make([]av1.DecoderEvent, 16*len(payloads)+64)
	probeSpans := make([]av1.TileSpan, av1.MaxTiles)
	probeJobs := make([]av1.TileJob, av1.MaxTiles)
	probeBatches := make([]av1.TileBatch, av1.MaxTiles)
	probeRTPBuffer := make([]byte, rtpBufferBytes)
	probeRTPSpans := make([]av1.RTPObuSpan, 16*len(payloads)+64)
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(
		probeStream, 0, payloads, workers, probeRTPBuffer, probeRTPSpans,
		probeEvents, probeSpans, probeJobs, probeBatches)
	if err != nil {
		t.Fatalf("RTP stream plan: %v", err)
	}
	if !plan.HasEvent() {
		t.Fatal("RTP stream plan did not identify a bind event")
	}

	format, err := av1.FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
	}
	const surfaceCount = av1.RefFrames + 1
	_, backingSize, err := av1.FramePoolRequiredSize(format, surfaceCount)
	if err != nil {
		t.Fatalf("FramePoolRequiredSize: %v", err)
	}
	pool, err := av1.BindFramePool(make([]byte, backingSize), format,
		make([]av1.Frame, surfaceCount), make([]int, surfaceCount), make([]bool, surfaceCount))
	if err != nil {
		t.Fatalf("BindFramePool: %v", err)
	}

	var (
		stream     av1.DecoderStream
		refs       av1.DecoderSurfaceReferences
		state      av1.DecoderFrameWorkState
		stats      av1.DecoderFrameWorkTileResidualStats
		sideData   av1.DecoderFrameWorkSideData
		batch      av1.DecoderFrameWorkBatchResidualRunner
		postFilter av1.DecoderFrameWorkReusableSupportedPostFilterRunner
	)
	scratch := lowLevelStreamScratch(plan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream,
		av1.DecoderFrameWorkResidualEventRuntime{
			State:             &state,
			Refs:              &refs,
			FramePool:         &pool,
			Align:             64,
			ReferenceSurfaces: make([]int, av1.InterRefsPerFrame),
			ReferenceFrames:   make([]*av1.Frame, av1.InterRefsPerFrame),
			Releases:          make([]int, av1.RefFrames),
			WorkerPool:        workerPool,
			SideData:          &sideData,
			Stats:             &stats,
		}, scratch, &batch)
	if err != nil {
		t.Fatalf("bind RTP runner: %v", err)
	}

	var result av1.DecoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadsIntoWithPostFilterRunner(&result, payloads, &postFilter); err != nil {
		t.Fatalf("RunRTPPayloadsIntoWithPostFilterRunner: %v", err)
	}
	var digests [][16]byte
	for _, f := range result.Run.Outputs {
		if f != nil {
			digests = append(digests, frameMD5Visible(f))
		}
	}
	return digests
}

func lowLevelStreamScratch(size av1.DecoderFrameWorkResidualStreamScratchSize) av1.DecoderFrameWorkResidualStreamScratch {
	return av1.DecoderFrameWorkResidualStreamScratch{
		Events:    make([]av1.DecoderEvent, size.Events),
		Event:     lowLevelEventScratch(size.Event),
		SideData:  lowLevelSideDataScratch(size.Event.SideData),
		Outputs:   make([]*av1.Frame, size.Event.Outputs),
		RTPBuffer: make([]byte, size.RTPBuffer),
		RTPSpans:  make([]av1.RTPObuSpan, size.RTPSpans),
	}
}

func lowLevelEventScratch(size av1.DecoderFrameWorkResidualEventScratchSize) av1.DecoderFrameWorkResidualEventScratch {
	return av1.DecoderFrameWorkResidualEventScratch{
		Runner:   lowLevelBatchRunnerScratch(size.Runner),
		SideData: lowLevelSideDataScratch(size.SideData),
		Spans:    make([]av1.TileSpan, size.Plan.SpanCount),
		Jobs:     make([]av1.TileJob, size.Plan.JobCount),
		Batches:  make([]av1.TileBatch, size.Plan.BatchCount),
	}
}

func lowLevelBatchRunnerScratch(size av1.DecoderFrameWorkBatchResidualRunnerScratchSize) av1.DecoderFrameWorkBatchResidualRunnerScratch {
	return av1.DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]av1.TileDecodeState, size.Workers),
		Storages:                make([]av1.DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]av1.DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]av1.DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]av1.DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]av1.DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]av1.DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func lowLevelSideDataScratch(size av1.DecoderFrameWorkSideDataScratchSize) av1.DecoderFrameWorkSideDataScratch {
	return av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap),
		CDEFReadMap:              make([]bool, size.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, size.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow),
	}
}

// frameMD5Visible reproduces the libaom per-frame digest layout: visible Y rows,
// then visible U rows, then visible V rows, each stripped of stride padding.
func frameMD5Visible(f *av1.Frame) [16]byte {
	h := md5.New()
	bps := f.Layout.BytesPerSample
	for _, plane := range []av1.FramePlane{f.Y, f.U, f.V} {
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * bps
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			h.Write(plane.Pix[off : off+rowBytes])
		}
	}
	var sum [16]byte
	h.Sum(sum[:0])
	return sum
}

// decodedFrameMD5 hashes a DecodedFrame's packed visible planes, which already
// have no stride padding.
func decodedFrameMD5(f av1.DecodedFrame) [16]byte {
	h := md5.New()
	h.Write(f.Y)
	h.Write(f.U)
	h.Write(f.V)
	var sum [16]byte
	h.Sum(sum[:0])
	return sum
}
