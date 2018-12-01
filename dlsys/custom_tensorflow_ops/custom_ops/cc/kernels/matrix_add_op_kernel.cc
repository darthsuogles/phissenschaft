#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "matrix_add_kernels.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class MatrixAddOp : public OpKernel {
   public:
    explicit MatrixAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& X = ctx->input(0);
        const Tensor& Y = ctx->input(1);

        if (!ctx->status().ok()) {
            return;
        }

        OP_REQUIRES(ctx, X.shape() == Y.shape(), errors::InvalidArgument("Input shapes have to be the same"));

        const int N = X.dim_size(0);
        const int H = X.dim_size(1);
        const int W = X.dim_size(2);
        const int C = X.dim_size(3);

        TensorShape output_shape({N, H, W, C});
        // same as: output_shape.AddDim(N); ....

        Tensor* Z = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &Z));
        // same as "OP_REQUIRES_OK(ctx,ctx->allocate_output(0, X.tensor<Dtype,
        // 4>().shape(), &Z));"

        ::tensorflow::functor::MatrixAddFunctor<Device, Dtype>::launch(ctx, X, Y, Z, bias_);
    }

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(MatrixAddOp);
    float bias_;
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class MatrixAddGradOp : public OpKernel {
   public:
    explicit MatrixAddGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& X = ctx->input(0);
        const Tensor& Y = ctx->input(1);
        const Tensor& topdiff = ctx->input(2);

        if (!ctx->status().ok()) {
            return;
        }

        Tensor* grad_X = nullptr;
        Tensor* grad_Y = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, X.shape(), &grad_X));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, Y.shape(), &grad_Y));

        ::tensorflow::functor::MatrixAddGrad<Device, Dtype>::launch(ctx, topdiff, grad_X, grad_Y);
    }
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T) \
    REGISTER_KERNEL_BUILDER(Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(MatrixAdd, CPU, uint32);
REGISTER_CUSTOM_OP(MatrixAdd, CPU, int32);
REGISTER_CUSTOM_OP(MatrixAdd, CPU, float);
REGISTER_CUSTOM_OP(MatrixAdd, CPU, double);
REGISTER_CUSTOM_OP(MatrixAddGrad, CPU, float);
REGISTER_CUSTOM_OP(MatrixAddGrad, CPU, double);

//#if GOOGLE_CUDA
REGISTER_CUSTOM_OP(MatrixAdd, GPU, uint32);
REGISTER_CUSTOM_OP(MatrixAdd, GPU, int32);
REGISTER_CUSTOM_OP(MatrixAdd, GPU, float);
REGISTER_CUSTOM_OP(MatrixAdd, GPU, double);
REGISTER_CUSTOM_OP(MatrixAddGrad, GPU, float);
REGISTER_CUSTOM_OP(MatrixAddGrad, GPU, double);
// #endif // GOOGLE_CUDA
#undef REGISTER_CUSTOM_OP

} // namespace tensorflow
