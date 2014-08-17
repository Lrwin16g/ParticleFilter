#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

// 状態モデルの仮想クラス
class StateModel
{
public:
    StateModel(double sigma);
    virtual ~StateModel();
    void translateState(const double *src, double *dst);
    inline int dimension() const {return dimension_;}
    
protected:
    int dimension_;
    double **stateTransMat_;
    double sigma_;
};

// 線形モデル
class LinearModel : public StateModel
{
public:
    LinearModel(double sigma);
};

// ランダムウォークモデル
class RandomModel : public StateModel
{
public:
    RandomModel(double sigma);
};

// パーティクルフィルタクラス
class ParticleFilter
{
public:
    ParticleFilter(int particleNum, StateModel *model);
    ~ParticleFilter();
    
    void initParticles(double *boundary[2]);
    void predict();
    void normalizeLikelihood();
    void estimate();
    void filterParticles();
    
    inline int dimension() const {return model_->dimension();}
    
    inline double particles(int row, int col) const
	{return particles_[row][col];}
    
    inline double likelihood(int index) const
	{return likelihood_[index];}
    
    inline void setLikelihood(int index, double value)
	{likelihood_[index] = value;}
    
    inline double estimateResult(int index) const
	{return estimateResult_[index];}
    
private:
    int particleNum_;
    double **particles_;
    double **newParticles_;
    double *likelihood_;
    double *estimateResult_;
    
    StateModel *model_;
    
    // DISALLOW_COPY_AND_ASSIGN
    ParticleFilter(const ParticleFilter&);
    void operator=(const ParticleFilter&);
};

#endif
