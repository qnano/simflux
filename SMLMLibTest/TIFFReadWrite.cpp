#include "CudaUtils.h"
#include <tiff/tiffio.h>
#include <tiff/tiff.h>
#include <Windows.h>

#include "TIFFReadWrite.h"



void WriteTIFF(const char *fn, const float* data, int width, int height, bool normalize)
{
	TIFF *tif = TIFFOpen(fn, "w");

	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, TIFF_BYTE);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

	TIFFSetField(tif, TIFFTAG_XRESOLUTION, 150.0f);
	TIFFSetField(tif, TIFFTAG_YRESOLUTION, 150.0f);
	TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);

	float scale = 1.0f, offset = 0.0f;
	if (normalize)
	{
		float minv, maxv;
		minv = maxv = data[0];
		for (int i = 0; i < width*height; i++)
		{
			if (minv > data[i]) minv = data[i];
			if (maxv < data[i]) maxv = data[i];
		}
		DebugPrintf("fn=%s, min=%f, max=%f\n", fn, minv, maxv);
		offset = -minv;
		if (minv != maxv)
			scale = 255.0f / (maxv - minv);
	}

	uint8_t* row = new uint8_t[width];

	for (int i = 0; i < height; i++) {
		for (int x = 0; x < width; x++) {
			float v = (data[i*width + x]+offset) * scale;
			if (v > 255.0f) v = 255.0f;
			row[x] = (uint8_t)v;
		}
		TIFFWriteScanline(tif, row, i, 0);
	}

	delete[] row;
	TIFFClose(tif);
}

void WriteTIFF(const char *fn, const DeviceArray<float>& data, int width, bool normalize)
{
	auto img = data.ToVector();
	WriteTIFF(fn, &img[0], width, int(img.size()) / width, normalize);
}

void WriteTIFF(const char *fn, const DeviceImage<float>& d_img, bool normalize)
{
	std::vector<float> img(d_img.NumPixels());
	d_img.CopyToHost(&img[0]);

	WriteTIFF(fn, &img[0], d_img.width, d_img.height,normalize);
}



void TIFFErrorCallback(const char* mdl, const char* fmt, va_list ap) {
	char buf[256];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);
	DebugPrintf("TIFF module %s. Error: %s\n", mdl, buf);
}
void TIFFWarningCallback(const char* mdl, const char* fmt, va_list ap) {
	char buf[256];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);
	DebugPrintf("TIFF module %s. Warning: %s\n", mdl, buf);
}

TIFFReader::TIFFReader(const char *fn)
{
	TIFFSetErrorHandler(TIFFErrorCallback);
	TIFFSetWarningHandler(TIFFWarningCallback);
	
	tif = TIFFOpen(fn, "r");
	hasNext = tif != 0;
	nImagesRead = 0;

	uint16_t bps;
	if (TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps) != 1) {
		DebugPrintf("tiff: no bps tag\n");
		TIFFClose(tif);
		tif = 0;
		return;
	}
	else
		bitsPerSample = bps;

	uint16_t channels;
	if (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels) != 1) {
		DebugPrintf("tiff: no samples per pixel tag\n");
		TIFFClose(tif);
		tif = 0;
		return;
	}
	else
		nChannels = channels;

	uint16_t fmt;
	if (TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &fmt) != 1)
		DebugPrintf("tiff: no sampleformat tag\n");
	//SAMPLEFORMAT_UINT

	int w, h;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
}


TIFFReader::~TIFFReader()
{
	if (tif)
		TIFFClose(tif);
}

bool TIFFReader::Read(std::vector<float>& dst, int& w, int& h)
{
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
	if (nImagesRead == 0) {

		const char *docname = 0;
		if (TIFFGetFieldDefaulted(tif, TIFFTAG_DOCUMENTNAME, &docname) == 1)
		{
			DebugPrintf("TIFF docname: %s\n", docname);
		}
		const char *imgdesc = 0;
		if (TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGEDESCRIPTION, &imgdesc) == 1)
		{
			DebugPrintf("TIFF imgdesc: %s\n", imgdesc);
		}
		DebugPrintf("TIFF width: %d, height: %d\n", w, h);
	}

		/*	int numTags = TIFFGetTagListCount(tif);
		for (int i = 0; i < numTags; i++) {
		uint32_t tag = TIFFGetTagListEntry(tif, i);
		const TIFFField* f = TIFFFieldWithTag(tif, tag);
		TIFFDataType dtype = TIFFFieldDataType(f);
		char data[256];

		switch (dtype) {
		case TIFF_BYTE: { uint8_t v; TIFFGetField(tif, tag, &v); sprintf(data, "%d", v); break;  }
		case TIFF_LONG: { uint32_t v; TIFFGetField(tif, tag, &v); sprintf(data, "%d", v); break;  }
		case TIFF_SLONG: { int32_t v; TIFFGetField(tif, tag, &v); sprintf(data, "%d", v); break; }
		case TIFF_SSHORT: { int16_t v; TIFFGetField(tif, tag, &v); sprintf(data, "%d", v); break; }
		case TIFF_SHORT: { uint16_t v; TIFFGetField(tif, tag, &v); sprintf(data, "%d", v); break; }
		case TIFF_FLOAT: { float v; TIFFGetField(tif, tag, &v); sprintf(data, "%f", v); break; }
		case TIFF_DOUBLE: { double v; TIFFGetField(tif, tag, &v); sprintf(data, "%f", v); break; }
		case TIFF_ASCII: { const char* str; TIFFGetField(tif, tag, &str); strncpy(data, str, sizeof(data)); break; }
		}

		DebugPrintf("TIFF Field %s\n", TIFFFieldName(f));
		}*/

	dst.resize(w*h);

	auto readPixels = [&](auto smp) {
		auto *row = new decltype(smp)[w];
		for (int y = 0; y < h; y++) {
			TIFFReadScanline(tif, row, y);
			for (int x = 0; x < w; x++)
				dst[y*h + x] = (float)row[x];
		}
		delete[] row;
	};
	if (bitsPerSample == 16)
		readPixels(uint16_t());
	else if (bitsPerSample == 8)
		readPixels(uint8_t());
	else if (bitsPerSample == 32)
		readPixels(uint32_t());

	nImagesRead++;

	bool next = TIFFReadDirectory(tif) != 0;
	return next;
}

