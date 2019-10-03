#pragma once

template<typename T>
class DeviceImage;

template<typename T>
class DeviceArray;

void WriteTIFF(const char *fn, const DeviceImage<float>& d_img, bool normalize=true);
void WriteTIFF(const char *fn, const float* data, int width, int height, bool normalize = true);
void WriteTIFF(const char *fn, const DeviceArray<float>& data,int width, bool normalize = true);

typedef struct tiff TIFF;

class TIFFReader {
public:
	TIFFReader(const char *fn);
	~TIFFReader();

	bool IsOpen() { return tif != 0;  }
	bool Read(std::vector<float>& dst, int& width, int& height); // returns false if no more images in the TIFF file

protected:
	TIFF * tif;
	bool hasNext;
	int nImagesRead;
	int bitsPerSample;
	int nChannels;
};

