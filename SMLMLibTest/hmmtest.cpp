#include "HMM.h"
#include "StringUtils.h"
#include "RandomDistributions.h"


void HMMViterbiTest()
{
	const int nstates = 3;
	typedef HMM<nstates> hmm;

	const int nsamples = 10;
	int states[nsamples];
	float samples[nsamples];

	float sigma = 0.2f;

	for (int i = 0; i < nsamples; i++)
	{
		states[i] = rand() % nstates;
		samples[i] = rand_normal<float>() * sigma + states[i];
	}
	DebugPrintf("Samples: \n");
	PrintVector(samples, nsamples);
	DebugPrintf("True states: \n");
	PrintVector(states, nsamples, " %d" );

	float priors[nstates];
	for (int i = 0; i < nstates; i++)
		priors[i] = 1.0f / nstates;

	float tr[nstates*nstates];
	float logtr[nstates*nstates];

	for (int i = 0; i < nstates; i++)
		for (int j = 0; j < nstates; j++) {
			tr[i*nstates + j] = 0.2f + rand_uniform<float>() * 0.6f;
			logtr[i*nstates + j] = log(tr[i*nstates + j]);
		}
	
	auto logEmissionProb = [&](int smp, int state) {
		return log(pdf_normal(samples[smp], state, sigma));
	};
	int output[nsamples];
	hmm::Viterbi(nsamples, priors, logtr, logEmissionProb, output);

	DebugPrintf("Transition matrix: \n");
	printMatrix(tr, nstates, nstates);

	DebugPrintf("Viterbi estimate: \n");
	PrintVector(output, nsamples, " %d");

	float posterior[nstates*nsamples];
	hmm::ForwardBackward(nsamples, priors, logtr, logEmissionProb, posterior);

	DebugPrintf("Posterior prob:\n");
	// logprob to prob
	for (int i = 0; i < nstates*nsamples; i++)
		posterior[i] = exp(posterior[i]);

	printMatrix(posterior, nsamples, nstates);
}

