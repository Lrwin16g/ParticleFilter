#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

// 状態モデルの基底クラス
class StateModel
{
public:
    StateModel(double sigma);
    virtual ~StateModel();
    virtual void translateState(const double *src, double *dst) = 0;
    inline int dimension() const {return dimension_;}
    
protected:
    int dimension_;
    double **stateTransMat_;
    double sigma_;
};

// 等速直線運動モデル
class LinearModel : public StateModel
{
public:
    LinearModel(double sigma);
    void translateState(const double *src, double *dst);
};

// 等加速度運動モデル
class AccelerateModel : public StateModel
{
public:
    AccelerateModel(double sigma);
    void translateState(const double *src, double *dst);
};

// ランダムウォークモデル
class RandomModel : public StateModel
{
public:
    RandomModel(double sigma);
    void translateState(const double *src, double *dst);
};

// パーティクルフィルタクラス
class ParticleFilter
{
public:
    ParticleFilter(int particleNum, StateModel *model);
    ~ParticleFilter();
    
    void initParticles(const double boundary[][2]);
    void predict();
    void normalizeLikelihood();
    void estimate();
    void resample();
    void resampleMultinomial();
    void resampleResidual();
    
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
    double *multidist_;
    
    StateModel *model_;
    
    // DISALLOW_COPY_AND_ASSIGN
    ParticleFilter(const ParticleFilter&);
    void operator=(const ParticleFilter&);
};

#endif
