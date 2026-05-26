package testvector

import (
	"bytes"
	"context"
	"fmt"
)

const (
	LibaomTestDataURL = "https://storage.googleapis.com/aom-test-data"
	libaomSource      = "libaom v3.14.0 test/test_vectors.cc and test/test-data.sha1"
)

var libaomRemoteVectors = [...]RemoteVector{
	{
		Tag:    TagDecoderLibaomQuantizer00,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit quantizer 00",
		Stream: RemoteFile{Name: "av1-1-b8-00-quantizer-00.ivf", SHA1: "c2e1ec9936b95254187a359e94aa32a9f3dad1b7"},
		MD5:    RemoteFile{Name: "av1-1-b8-00-quantizer-00.ivf.md5", SHA1: "26cd2a0321d01d9db5f6dace8b43a40cd5b9d58d"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelQuantizer,
	},
	{
		Tag:    TagDecoderLibaomQuantizer01,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit quantizer 01",
		Stream: RemoteFile{Name: "av1-1-b8-00-quantizer-01.ivf", SHA1: "a56dd02c0258d4afea1ee358a22b54e99e39d5e1"},
		MD5:    RemoteFile{Name: "av1-1-b8-00-quantizer-01.ivf.md5", SHA1: "b3d24124d81f1fbb26f5eb0036accb54f3ec69b2"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelQuantizer,
	},
	{
		Tag:    TagDecoderLibaomSize16x16,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit 16x16 size",
		Stream: RemoteFile{Name: "av1-1-b8-01-size-16x16.ivf", SHA1: "838388fbda4a1d91be81ff62694c3bf13c460d38"},
		MD5:    RemoteFile{Name: "av1-1-b8-01-size-16x16.ivf.md5", SHA1: "4229c1caf8e25eb3073456fb90ceed206753901e"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelSize,
	},
	{
		Tag:    TagDecoderLibaomQuantizer10Bit00,
		Kind:   KindDecoder,
		Name:   "libaom av1 10-bit quantizer 00",
		Stream: RemoteFile{Name: "av1-1-b10-00-quantizer-00.ivf", SHA1: "9bbe8499796aa588ff02e313fb0d4349940d2fea"},
		MD5:    RemoteFile{Name: "av1-1-b10-00-quantizer-00.ivf.md5", SHA1: "36b402eedad2bacee8ac09acce44e2fc356dd80b"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelQuantizer | VectorLabelHighBitDepth,
	},
	{
		Tag:    TagDecoderLibaomAllIntra,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit all-intra",
		Stream: RemoteFile{Name: "av1-1-b8-02-allintra.ivf", SHA1: "a9f7ea6312a533cc6426a6145edd190d45813c37"},
		MD5:    RemoteFile{Name: "av1-1-b8-02-allintra.ivf.md5", SHA1: "8fd8f789cfee1069d20f3e2c241f5cad7292239e"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelIntra,
	},
	{
		Tag:    TagDecoderLibaomCDFUpdate,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit cdf update",
		Stream: RemoteFile{Name: "av1-1-b8-04-cdfupdate.ivf", SHA1: "afca5502a489692b0a3c120370b0f43b8fc572a1"},
		MD5:    RemoteFile{Name: "av1-1-b8-04-cdfupdate.ivf.md5", SHA1: "13b9423155a08d5e3a2fd9ae4a973bb046718cdf"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelCDF,
	},
	{
		Tag:    TagDecoderLibaomMV,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit mv",
		Stream: RemoteFile{Name: "av1-1-b8-05-mv.ivf", SHA1: "f064290d7fcd3b3de19020e8aec6c43c88d3a505"},
		MD5:    RemoteFile{Name: "av1-1-b8-05-mv.ivf.md5", SHA1: "bff316e63ded5559116bdc2fa4aa97ad7b1a1761"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelMotion,
	},
	{
		Tag:    TagDecoderLibaomMFMV,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit mfmv",
		Stream: RemoteFile{Name: "av1-1-b8-06-mfmv.ivf", SHA1: "b48a717c7c003b8dd23c3c2caed1ac673380fdb3"},
		MD5:    RemoteFile{Name: "av1-1-b8-06-mfmv.ivf.md5", SHA1: "1424e3cb53e00eb56b94f4c725826274212c42b6"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelMFMV | VectorLabelMotion,
	},
	{
		Tag:    TagDecoderLibaomIntrabc,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit intra-only intrabc extreme dv",
		Stream: RemoteFile{Name: "av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf", SHA1: "c7f336958e7af6162c20ddc84d67c7dfa9826910"},
		MD5:    RemoteFile{Name: "av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf.md5", SHA1: "36a4fcf07e645ed522cde5845dd9c6ab2b2d1502"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelIntra | VectorLabelIntrabc,
	},
	{
		Tag:    TagDecoderLibaomSVC,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit svc L1T2",
		Stream: RemoteFile{Name: "av1-1-b8-22-svc-L1T2.ivf", SHA1: "e94687eb0e90179b3800b6d5e11eb7e9bfb34eec"},
		MD5:    RemoteFile{Name: "av1-1-b8-22-svc-L1T2.ivf.md5", SHA1: "2bc12b16385ea14323bc79607fb8dfbd7edaf8ef"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelSVC,
	},
	{
		Tag:    TagDecoderLibaomFilmGrain,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit film grain 50",
		Stream: RemoteFile{Name: "av1-1-b8-23-film_grain-50.ivf", SHA1: "11bb40026103182c23a88133edafca369e5575e2"},
		MD5:    RemoteFile{Name: "av1-1-b8-23-film_grain-50.ivf.md5", SHA1: "c58ccf7ff04711acc559c06f0bfce3c5b14800c3"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelFilmGrain,
	},
	{
		Tag:    TagDecoderLibaomMonochrome,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit monochrome",
		Stream: RemoteFile{Name: "av1-1-b8-24-monochrome.ivf", SHA1: "a17584012187cd886b64f8cb0f35bfd8d762f9dc"},
		MD5:    RemoteFile{Name: "av1-1-b8-24-monochrome.ivf.md5", SHA1: "e71cd9a07f928c527c900daddd071ae60337426d"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelFast | SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelMonochrome,
	},
	{
		Tag:    TagDecoderLibaomFilmGrain10Bit,
		Kind:   KindDecoder,
		Name:   "libaom av1 10-bit film grain 50",
		Stream: RemoteFile{Name: "av1-1-b10-23-film_grain-50.ivf", SHA1: "2f883c7e11c21a31f79bd9c809541be90b0c7c4a"},
		MD5:    RemoteFile{Name: "av1-1-b10-23-film_grain-50.ivf.md5", SHA1: "83f2094fca597ad38b4fd623b807de1774c53ffb"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelFilmGrain | VectorLabelHighBitDepth,
	},
	{
		Tag:    TagDecoderLibaomMonochrome10Bit,
		Kind:   KindDecoder,
		Name:   "libaom av1 10-bit monochrome",
		Stream: RemoteFile{Name: "av1-1-b10-24-monochrome.ivf", SHA1: "03a8d002594ccc51932332002bb6f9837ef46d0f"},
		MD5:    RemoteFile{Name: "av1-1-b10-24-monochrome.ivf.md5", SHA1: "e24aa6951afd7b2bb53eb1a73e25a19e7b189f82"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelRelevant | SuiteLevelFull,
		Labels: VectorLabelMonochrome | VectorLabelHighBitDepth,
	},

	// Extended cohort. These vectors are tagged SuiteLevelExtended |
	// SuiteLevelFull so they are downloaded by testvectors-full but stay
	// out of the fast-suite default. They are intended to probe latent
	// gaps that the small fast cohort cannot surface:
	//   - mid- and lossless-Q 10-bit quantizer streams
	//   - sub-block-aligned and odd 34x34 / 66x66 sizes
	//   - >= 192x192 resolutions that hit multi-superblock tile rows
	//   - the remaining libaom SVC permutations (L2T1, L2T2)
	{
		Tag:    TagDecoderLibaomQuantizer10Bit32,
		Kind:   KindDecoder,
		Name:   "libaom av1 10-bit quantizer 32",
		Stream: RemoteFile{Name: "av1-1-b10-00-quantizer-32.ivf", SHA1: "96291f6ace85980892d135a5b74188cd629c325f"},
		MD5:    RemoteFile{Name: "av1-1-b10-00-quantizer-32.ivf.md5", SHA1: "a5ceace390d4a75d48281fe29060c21557e4f5ae"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelQuantizer | VectorLabelHighBitDepth,
	},
	{
		Tag:    TagDecoderLibaomQuantizer10Bit63,
		Kind:   KindDecoder,
		Name:   "libaom av1 10-bit quantizer 63",
		Stream: RemoteFile{Name: "av1-1-b10-00-quantizer-63.ivf", SHA1: "8b6eb3fff2e0db7eac775b08c745250ca591e2d9"},
		MD5:    RemoteFile{Name: "av1-1-b10-00-quantizer-63.ivf.md5", SHA1: "63ea689d025593e5d91760785b8e446d04d4671e"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelQuantizer | VectorLabelHighBitDepth,
	},
	{
		Tag:    TagDecoderLibaomSize34x34,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit 34x34 size",
		Stream: RemoteFile{Name: "av1-1-b8-01-size-34x34.ivf", SHA1: "4cc2a437dba56e8878113d9b390b980522542028"},
		MD5:    RemoteFile{Name: "av1-1-b8-01-size-34x34.ivf.md5", SHA1: "57eeb971f00e64abde25be69dbcb4e3ce5065a57"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSize,
	},
	{
		Tag:    TagDecoderLibaomSize64x64,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit 64x64 size",
		Stream: RemoteFile{Name: "av1-1-b8-01-size-64x64.ivf", SHA1: "7d017daebef2d39ed42a505a8e6103ab0c0988c1"},
		MD5:    RemoteFile{Name: "av1-1-b8-01-size-64x64.ivf.md5", SHA1: "1ff38d5ecba82fb2e6ac3b09c29c9fe74885ac29"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSize,
	},
	{
		Tag:    TagDecoderLibaomSize66x66,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit 66x66 size",
		Stream: RemoteFile{Name: "av1-1-b8-01-size-66x66.ivf", SHA1: "68b47b386f61d67cb5b824a7e6bf87c8b9c2bf7b"},
		MD5:    RemoteFile{Name: "av1-1-b8-01-size-66x66.ivf.md5", SHA1: "25974893956ebd92df474325946130c34f880ea7"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSize,
	},
	{
		Tag:    TagDecoderLibaomSize208x208,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit 208x208 size",
		Stream: RemoteFile{Name: "av1-1-b8-01-size-208x208.ivf", SHA1: "e5900d9150c8bebc49776227afd3b0a21f5a6ac6"},
		MD5:    RemoteFile{Name: "av1-1-b8-01-size-208x208.ivf.md5", SHA1: "86d6b9a3840aa0a77938547c905bd6f45d069681"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSize,
	},
	{
		Tag:    TagDecoderLibaomSize226x226,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit 226x226 size",
		Stream: RemoteFile{Name: "av1-1-b8-01-size-226x226.ivf", SHA1: "40dd208eb525cd90d7c0674cf787097fb909afae"},
		MD5:    RemoteFile{Name: "av1-1-b8-01-size-226x226.ivf.md5", SHA1: "34bdef682a4eae0e0a05e4486a968af1df8b220a"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSize,
	},
	{
		Tag:    TagDecoderLibaomSVCL2T1,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit svc L2T1",
		Stream: RemoteFile{Name: "av1-1-b8-22-svc-L2T1.ivf", SHA1: "e14825f50ff845b8a6932c64cb254007a0b5e3a1"},
		MD5:    RemoteFile{Name: "av1-1-b8-22-svc-L2T1.ivf.md5", SHA1: "0f75f2ac44e61fc83be70c955410fa378e433237"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSVC,
	},
	{
		Tag:    TagDecoderLibaomSVCL2T2,
		Kind:   KindDecoder,
		Name:   "libaom av1 8-bit svc L2T2",
		Stream: RemoteFile{Name: "av1-1-b8-22-svc-L2T2.ivf", SHA1: "32ef2f14ee9cb11a24a22934f4c065e926e5d236"},
		MD5:    RemoteFile{Name: "av1-1-b8-22-svc-L2T2.ivf.md5", SHA1: "f476a10ff06d750129f8229755d51e17ff141b2a"},
		Oracle: OracleFrameMD5,
		Levels: SuiteLevelExtended | SuiteLevelFull,
		Labels: VectorLabelSVC,
	},
}

func LibaomRemoteManifest() RemoteManifest {
	return RemoteManifest{
		Name:    "libaom-v3.14.0-av1",
		Source:  libaomSource,
		BaseURL: LibaomTestDataURL,
		Vectors: libaomRemoteVectors[:],
	}
}

func LibaomRemoteVector(tag Tag) (RemoteVector, bool) {
	return LibaomRemoteManifest().FindRemote(tag)
}

func LibaomRemoteSuite(ctx context.Context, cacheDir string, level SuiteLevel, labels VectorLabel) (Suite, error) {
	return LoadRemoteSuite(ctx, LibaomRemoteManifest(), cacheDir, level, labels)
}

func ParseLibaomMD5Digests(tag Tag, src []byte) ([]FrameDigest, error) {
	lines := bytes.Split(bytes.TrimSpace(src), []byte{'\n'})
	digests := make([]FrameDigest, 0, len(lines))
	for i, line := range lines {
		fields := bytes.Fields(line)
		if len(fields) == 0 {
			continue
		}
		md5, err := ParseMD5Hex(fields[0])
		if err != nil {
			return nil, fmt.Errorf("md5 line %d: %w", i, err)
		}
		digests = append(digests, FrameDigest{
			Tag:        tag,
			FrameIndex: uint32(len(digests)),
			MD5:        md5,
		})
	}
	if len(digests) == 0 {
		return nil, ErrMissingDigest
	}
	return digests, nil
}
