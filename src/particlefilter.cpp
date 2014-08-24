#include "particlefilter.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "statlib.h"

StateModel::StateModel(double sigma)
    : sigma_(sigma), stateTransMat_(NULL)
{
}

StateModel::~StateModel()
{
    if (stateTransMat_ != NULL) {
	for (int i = 0; i < dimension_; ++i) {
	    delete[] stateTransMat_[i];
	}
	delete[] stateTransMat_;
    }
}

LinearModel::LinearModel(double sigma)
    : StateModel(sigma)
{
    dimension_ = 4;
    stateTransMat_ = new double*[dimension_];
    for (int i = 0; i < dimension_; ++i) {
	stateTransMat_[i] = new double[dimension_];
    }
    stateTransMat_[0][0] = 1.0; stateTransMat_[0][1] = 0.0;
    stateTransMat_[0][2] = 1.0; stateTransMat_[0][3] = 0.0;
    stateTransMat_[1][0] = 0.0; stateTransMat_[1][1] = 1.0;
    stateTransMat_[1][2] = 0.0; stateTransMat_[1][3] = 1.0;
    stateTransMat_[2][0] = 0.0; stateTransMat_[2][1] = 0.0;
    stateTransMat_[2][2] = 1.0; stateTransMat_[2][3] = 0.0;
    stateTransMat_[3][0] = 0.0; stateTransMat_[3][1] = 0.0;
    stateTransMat_[3][2] = 0.0; stateTransMat_[3][3] = 1.0;
}

void LinearModel::translateState(const double *src, double *dst)
{
    for (int i = 0; i < dimension_; ++i) {
	dst[i] = 0.0;
	for (int j = 0; j < dimension_; ++j) {
	    dst[i] += stateTransMat_[i][j] * src[j];
	}
    }
    dst[2] += stat::randn(0.0, sigma_);
    dst[3] += stat::randn(0.0, sigma_);
}

AccelerateModel::AccelerateModel(double sigma)
    : StateModel(sigma)
{
    dimension_ = 6;
    stateTransMat_ = new double*[dimension_];
    for (int i = 0; i < dimension_; ++i) {
	stateTransMat_[i] = new double[dimension_];
    }
    stateTransMat_[0][0] = 1.0; stateTransMat_[0][1] = 0.0; stateTransMat_[0][2] = 1.0;
    stateTransMat_[0][3] = 0.0; stateTransMat_[0][4] = 0.0; stateTransMat_[0][5] = 0.0;
    stateTransMat_[1][0] = 0.0; stateTransMat_[1][1] = 1.0; stateTransMat_[1][2] = 0.0;
    stateTransMat_[1][3] = 1.0; stateTransMat_[1][4] = 0.0; stateTransMat_[1][5] = 0.0;
    stateTransMat_[2][0] = 0.0; stateTransMat_[2][1] = 0.0; stateTransMat_[2][2] = 1.0;
    stateTransMat_[2][3] = 0.0; stateTransMat_[2][4] = 1.0; stateTransMat_[2][5] = 0.0;
    stateTransMat_[3][0] = 0.0; stateTransMat_[3][1] = 0.0; stateTransMat_[3][2] = 0.0;
    stateTransMat_[3][3] = 1.0; stateTransMat_[3][4] = 0.0; stateTransMat_[3][5] = 1.0;
    stateTransMat_[4][0] = 0.0; stateTransMat_[4][1] = 0.0; stateTransMat_[4][2] = 0.0;
    stateTransMat_[4][3] = 0.0; stateTransMat_[4][4] = 1.0; stateTransMat_[4][5] = 0.0;
    stateTransMat_[5][0] = 0.0; stateTransMat_[5][1] = 0.0; stateTransMat_[5][2] = 0.0;
    stateTransMat_[5][3] = 0.0; stateTransMat_[5][4] = 0.0; stateTransMat_[5][5] = 1.0;
}

void AccelerateModel::translateState(const double *src, double *dst)
{
    for (int i = 0; i < dimension_; ++i) {
	dst[i] = 0.0;
	for (int j = 0; j < dimension_; ++j) {
	    dst[i] += stateTransMat_[i][j] * src[j];
	}
    }
    dst[4] += stat::randn(0.0, sigma_);
    dst[5] += stat::randn(0.0, sigma_);
}

RandomModel::RandomModel(double sigma)
    : StateModel(sigma)
{
    dimension_ = 2;
    stateTransMat_ = new double*[dimension_];
    for (int i = 0; i < dimension_; ++i) {
	stateTransMat_[i] = new double[dimension_];
    }
    stateTransMat_[0][0] = 1.0; stateTransMat_[0][1] = 0.0;
    stateTransMat_[1][0] = 0.0; stateTransMat_[1][1] = 1.0;
}

void RandomModel::translateState(const double *src, double *dst)
{
    for (int i = 0; i < dimension_; ++i) {
	dst[i] = stat::randn(0.0, sigma_);
	for (int j = 0; j < dimension_; ++j) {
	    dst[i] += stateTransMat_[i][j] * src[j];
	}
    }
}

ParticleFilter::ParticleFilter(int particleNum, StateModel *model)
    : particleNum_(particleNum), model_(model), particles_(NULL),
      newParticles_(NULL), likelihood_(NULL), estimateResult_(NULL),
      multidist_(NULL)
{
    srand(time(NULL));
    
    particles_ = new double*[particleNum];
    newParticles_ = new double*[particleNum];
    for (int i = 0; i < particleNum_; ++i) {
	particles_[i] = new double[model_->dimension()];
	newParticles_[i] = new double[model_->dimension()];
    }

    likelihood_ = new double[particleNum_];
    estimateResult_ = new double[model_->dimension()];
    multidist_ = new double[particleNum_];
}

ParticleFilter::~ParticleFilter()
{
    if (particles_ != NULL) {
	for (int i = 0; i < particleNum_; ++i) {
	    delete[] particles_[i];
	}
	delete[] particles_;
    }
    if (newParticles_ != NULL) {
	for (int i = 0; i < particleNum_; ++i) {
	    delete[] newParticles_[i];
	}
	delete[] newParticles_;
    }
    if (likelihood_ != NULL) {
	delete[] likelihood_;
    }
    if (estimateResult_ != NULL) {
	delete[] estimateResult_;
    }
    if (multidist_ != NULL) {
	delete[] multidist_;
    }
}

// サンプルの初期化
void ParticleFilter::initParticles(double boundary[][2])
{
    for (int i = 0; i < particleNum_; ++i) {
	for (int j = 0; j < model_->dimension(); ++j) {
	    particles_[i][j] = stat::randu(boundary[j][0], boundary[j][1]);
	    newParticles_[i][j] = particles_[i][j];
	}
    }
}

// 予測
void ParticleFilter::predict()
{
    for (int i = 0; i < particleNum_; ++i) {
	model_->translateState(newParticles_[i], particles_[i]);
    }
}

// 尤度の正規化
void ParticleFilter::normalizeLikelihood()
{
    double sum = 0.0;
    for (int i = 0; i < particleNum_; ++i) {
	sum += likelihood_[i];
    }
    for (int i = 0; i < particleNum_; ++i) {
	likelihood_[i] /= sum;
    }
}

// 結果の推定
void ParticleFilter::estimate()
{
    for (int i = 0; i < model_->dimension(); ++i) {
	estimateResult_[i] = 0.0;
	for (int j = 0; j < particleNum_; ++j) {
	    estimateResult_[i] += likelihood_[j] * particles_[j][i];
	}
    }
}

// リサンプリング
void ParticleFilter::resample()
{
    double maxLikelihood = 0.0;
    int maxIndex = 0;
    int count = 0, restoreCount = 0;
    for (int i = 0; i < particleNum_; ++i) {
	// リサンプリングの数がパーティクル数より少なかった場合用に最も尤度の高いサンプルを求める
	if (maxLikelihood < likelihood_[i]) {
	    maxLikelihood = likelihood_[i];
	    maxIndex = i;
	}
	// 尤度に応じて複製するサンプルの数を求める
	int restoreNum = floor(likelihood_[i] * particleNum_ + 0.5);
	restoreCount += restoreNum;
	// リサンプリングの数がパーティクル数を超えた場合、打ち切り
	if (restoreCount >= particleNum_) {
	    int rest = particleNum_ - count - 1;
	    for (int j = 0; j < rest; ++j) {
		for (int k = 0; k < model_->dimension(); ++k) {
		    newParticles_[count][k] = particles_[i][k];
		}
		count++;
	    }
	    break;
	}
	// リサンプルの数だけサンプルを複製
	for (int j = 0; j < restoreNum; ++j) {
	    for (int k = 0; k < model_->dimension(); ++k) {
		newParticles_[count][k] = particles_[i][k];
	    }
	    count++;
	}
    }
    
    // リサンプリングの数がパーティクル数より少なかった場合、最も尤度が高いサンプルで埋める
    if (count < particleNum_ - 1) {
	int rest = particleNum_ - count;
	for (int i = 0; i < rest; ++i) {
	    for (int j = 0; j < model_->dimension(); ++j) {
		newParticles_[count + i][j] = particles_[maxIndex][j];
	    }
	}
    }
}

// 多項分布に従ってリサンプリング
void ParticleFilter::resampleMultinomial()
{
    multidist_[0] = likelihood_[0];
    for (int i = 1; i < particleNum_; ++i) {
	multidist_[i] = multidist_[i - 1] + likelihood_[i];
    }
    
    for (int i = 0; i < particleNum_; ++i) {
	double prob = stat::randu(0.0, 1.0);
	int index = 0;
	while (multidist_[++index] < prob) {
	    ;
	}
	for (int j = 0; j < model_->dimension(); ++j) {
	    newParticles_[i][j] = particles_[index][j];
	}
    }
}

void ParticleFilter::resampleResidual()
{
    double accum = 0.0;
    int count = 0;
    for (int i = 0; i < particleNum_; ++i) {
	accum += likelihood_[i];
	multidist_[i] = accum;
	int restoreNum = floor(likelihood_[i] * particleNum_);
	for (int j = 0; j < restoreNum; ++j) {
	    for (int k = 0; k < model_->dimension(); ++k) {
		newParticles_[count][k] = particles_[i][k];
	    }
	    count++;
	}
    }
    
    for (int i = count; i < particleNum_; ++i) {
	double prob = stat::randu(0.0, 1.0);
	int index = 0;
	while (multidist_[++index] < prob) {
	    ;
	}
	for (int j = 0; j < model_->dimension(); ++j) {
	    newParticles_[i][j] = particles_[index][j];
	}
    }
}
