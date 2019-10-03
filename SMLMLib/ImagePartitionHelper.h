#pragma once

template<typename T>
struct IndexWithValue {
	int index;
	T value;
};

struct NonNegativeIndex
{
	template<typename T>
	__host__ __device__
		bool operator()(const IndexWithValue<T> &x)
	{
		return x.index >= 0;
	}
};




template<typename T>
class ImagePartitionHelper
{
public:
	DeviceArray<IndexWithValue<T>> indices;
	DeviceArray<IndexWithValue<T>> selected;
	PinnedArray<IndexWithValue<T>> h_selected;
	DeviceArray<uint8_t> partitionTempStorage; // for cub::DevicePartition::If
	PinnedArray<int> h_count;
	DeviceArray<int> d_count;
	int2 imgsize;

	ImagePartitionHelper(int2 imgsize) : h_count(1), d_count(1), imgsize(imgsize),
		indices(imgsize.x*imgsize.y), selected(imgsize.x*imgsize.y), h_selected(imgsize.x*imgsize.y)
	{
		size_t tempBytes;
		CUDAErrorCheck(cub::DevicePartition::If(0, tempBytes, indices.ptr(), selected.ptr(),
			d_count.ptr(), (int)indices.size(), NonNegativeIndex(), 0));
		partitionTempStorage.Init(tempBytes);
	}

	template<typename Op>
	void ConvertToArray(const DeviceImage<T>& input, cudaStream_t stream, Op op)
	{
		auto input_ = input.GetConstIndexer();
		IndexWithValue<T>* dst = indices.data();
		LaunchKernel(input.height, input.width, [=]__device__(int y, int x) {
			T v = input_(x, y);
			if (op(v))
				dst[y*input_.width + x] = { y*input_.width + x,v };
			else
				dst[y*input_.width + x] = { -1,0 };
		}, 0, stream);
	}

	template<typename Op>
	void Select(DeviceImage<T>& image, cudaStream_t stream, Op op)
	{
		ConvertToArray(image, stream, op);
		size_t tmpsize = partitionTempStorage.size();
		CUDAErrorCheck(cub::DevicePartition::If(partitionTempStorage.ptr(), tmpsize, indices.ptr(),
			selected.ptr(), d_count.ptr(), (int)indices.size(), NonNegativeIndex(), stream));

		h_selected.CopyFromDevice(selected, stream);
		d_count.CopyToHost(h_count.data(), true, stream);
	}

	std::vector<IndexWithValue<T>> GetResults()
	{
		std::vector<IndexWithValue<T>> results(h_count[0]);
		for (int i = 0; i < h_count[0]; i++)
			results[i] = h_selected[i];
		return results;
	}

	void DebugPrint(const char *name)
	{
		for (auto r : GetResults()) {
			int y = r.index / imgsize.x;
			int x = r.index % imgsize.x;
			DebugPrintf("%s: %d,%d: %f\n", name, x, y,  (float) r.intensity);
		}
	}
};

