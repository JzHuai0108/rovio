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

TEST(EigenQuaternion, FromTwoVectors) {
  Eigen::Vector3d aW(0.1, -0.2, -9.8);
  Eigen::Vector3d bB(-0.135870944, -8.32164642, -3.26329686);
  Eigen::Quaterniond q_BW = q_BW.FromTwoVectors(aW, bB);
  aW.normalize();
  bB.normalize();
  EXPECT_TRUE(bB.isApprox(q_BW * aW, 1e-8));
  
  Eigen::Vector3d aPerp(1.0, 0, 0);
  Eigen::Vector3d rv = LWF::NormalVectorElement::getRotationFromTwoNormals(aW, bB, aPerp);
  Eigen::AngleAxisd aa_BW(rv.norm(), rv.normalized());
  EXPECT_TRUE(bB.isApprox(aa_BW * aW, 1e-8));
}

TEST(NormalVectorElement, RotationFromTwoNormals) {
  Eigen::Vector3d mu(0, 0, 1);
  Eigen::Vector3d nu(0.1, -0.1, 0.95);
  Eigen::Vector3d muPerp(1.0, 0, 0);
  Eigen::Vector3d rv = LWF::NormalVectorElement::getRotationFromTwoNormals(mu, nu, muPerp);
  Eigen::Vector3d rvTrue(0.224549, 0.224549, 0);
  EXPECT_TRUE(rv.isApprox(rvTrue, 1e-6));
}

TEST(RotationQuaternion, rotateAndOperatorDot) {
  // x y z w
  Eigen::Matrix<double, 4, 1> qc1 = Eigen::Matrix<double, 4, 1>::Random();
  qc1.normalize();
  QPD qpd1(qc1[3], qc1[0], qc1[1], qc1[2]);
  Eigen::Quaterniond q1(qc1);
  Eigen::Matrix<double, 3, 1> vec = Eigen::Matrix<double, 3, 1>::Random();
  EXPECT_TRUE((q1* vec).isApprox(qpd1.rotate(vec), 1e-8));

  Eigen::Matrix<double, 4, 1> qc2 = Eigen::Matrix<double, 4, 1>::Random();
  qc2.normalize();
  QPD qpd2(qc2[3], qc2[0], qc2[1], qc2[2]);
  Eigen::Quaterniond q2(qc2);
  EXPECT_TRUE((q1 * q2).coeffs().isApprox((qpd1 * qpd2).toImplementation().coeffs(), 1e-8));
}

TEST(RotationQuaternion, exponentialMap) {
  Eigen::Matrix<double, 3, 1> rv = Eigen::Matrix<double, 3, 1>::Random();
  Eigen::AngleAxisd aa(rv.norm(), rv.normalized());
  Eigen::Quaterniond quat(aa);
  QPD qpd = qpd.exponentialMap(rv);
  EXPECT_TRUE(qpd.toImplementation().coeffs().isApprox(quat.coeffs(), 1e-8));
}
