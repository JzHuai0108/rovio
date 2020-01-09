#include <memory>
#include "gtest/gtest.h"
#include <Eigen/StdVector>

#include "rovio/RovioFilter.hpp"
#include "rovio/RovioNode.hpp"
#ifdef MAKE_SCENE
#include "rovio/RovioScene.hpp"
#endif

class RovioManifoldTest : public virtual ::testing::Test {
 protected:
  // The below entities are computed with the online tool
  // https://www.andre-gaschler.com/rotationconverter/.
  RovioManifoldTest() : 
    rv_(0.02, 0.03, -0.04),
    aa_ref_(0.0538516, Eigen::Vector3d(0.3713907, 0.557086, -0.7427814)),
    quat_ref_(0.9996375, 0.0099988, 0.0149982, -0.0199976),
    eps(1e-6) {
    rm_ref_ <<   0.9987503,  0.0402806,  0.0295856,
    -0.0396807,  0.9990003, -0.0205902,
    -0.0303854,  0.0193905,  0.9993501;
  }
  virtual ~RovioManifoldTest() {
  }
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  Eigen::Vector3d rv_;
  Eigen::AngleAxis<double> aa_ref_;
  Eigen::Quaterniond quat_ref_;
  Eigen::Matrix3d rm_ref_;
  double eps;
};

TEST_F(RovioManifoldTest, Skew) {
  M3D skew_rv;
  skew_rv << 0, 0.04, 0.03, 
  -0.04, 0, -0.02, 
  -0.03, 0.02, 0;
  EXPECT_TRUE(gSM(rv_).isApprox(skew_rv, 1e-8));
}

TEST_F(RovioManifoldTest, EigenQuaternion) {
  Eigen::Quaterniond quat(aa_ref_);
  Eigen::Matrix<double, 4, 1> xyzw = quat.coeffs();
  Eigen::Matrix<double, 4, 1> xyzw_ref = quat_ref_.coeffs();
  EXPECT_TRUE(xyzw.isApprox(xyzw_ref, eps));
}

TEST_F(RovioManifoldTest, EigenRotation) {
  EXPECT_TRUE(aa_ref_.toRotationMatrix().isApprox(rm_ref_, eps));
  EXPECT_TRUE(quat_ref_.toRotationMatrix().isApprox(rm_ref_, eps));
}

TEST_F(RovioManifoldTest, kindrRotationQuaternion) {
  QPD q = q.exponentialMap(rv_);
  EXPECT_NEAR(q.vector()[0], quat_ref_.coeffs()[3], eps);
  EXPECT_TRUE(q.vector().tail<3>().isApprox(quat_ref_.coeffs().head<3>(), eps));
}

TEST_F(RovioManifoldTest, kindrRotationMatrix) {
  QPD q = q.exponentialMap(rv_);
  kindr::RotationMatrixD R1(q);
  EXPECT_TRUE(R1.matrix().isApprox(rm_ref_, eps));

  kindr::RotationVectorD rv(rv_);
  kindr::RotationMatrixD R2(rv);
  EXPECT_TRUE(R2.matrix().isApprox(rm_ref_, eps));

  kindr::RotationMatrixD R3 = R3.exponentialMap(rv_);
  EXPECT_TRUE(R3.matrix().isApprox(rm_ref_, eps));
}

// The fixture for testing class NormalVectorElementTest
class NormalVectorElementTesting : public virtual ::testing::Test {
 protected:
  NormalVectorElementTesting() {
    unsigned int s = 1;
    testElement1_.setRandom(s);
    testElement2_.setRandom(s);
  }
  virtual ~NormalVectorElementTesting() {
  }
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:
  LWF::NormalVectorElement testElement1_;
  LWF::NormalVectorElement testElement2_;
  LWF::NormalVectorElement testElement3_;
  LWF::NormalVectorElement::mtDifVec difVec_;
};

// Test derivative of vector
TEST_F(NormalVectorElementTesting, boxPlusDerivative) {
  const double d  = 1e-6;
  difVec_.setZero();
  difVec_(0) = d;
  testElement1_.boxPlus(difVec_,testElement2_);
  ASSERT_NEAR(((testElement2_.getVec()-testElement1_.getVec())/d-testElement1_.getM().col(0)).norm(),0.0,1e-6);
  difVec_.setZero();
  difVec_(1) = d;
  testElement1_.boxPlus(difVec_,testElement2_);
  ASSERT_NEAR(((testElement2_.getVec()-testElement1_.getVec())/d-testElement1_.getM().col(1)).norm(),0.0,1e-6);

  Eigen::Matrix<double, 3, 2> Mmu, Nmu;
  Mmu = testElement1_.getM();
  Nmu = testElement1_.getN();
  Eigen::Matrix<double, 3, 2> Manal = gSM(-testElement1_.getVec()) * Nmu;
  EXPECT_TRUE(Mmu.isApprox(Manal, 1e-6));
}
