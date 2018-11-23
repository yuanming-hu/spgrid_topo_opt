#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "Elasticity_Solver_Data.h"
#include "../topo_opt/fem_interface.h"
#include "CG/CG_VECTOR.h"
#include "CG/CG_SYSTEM.h"
#include "CG/CONJUGATE_GRADIENT.h"
#include "CPU_Kernels/Validate_Blocks.h"
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <limits>

using namespace fem_interface;
using namespace SPGrid;

struct SolverState {
  void *spgrid_ptr;
  double initial_residual_linf;

  SolverState() {
    spgrid_ptr = nullptr;
    initial_residual_linf = 0;
  }
};
extern "C" void fem_solve(void *input, void *output) {
#ifndef USING_DOUBLE
    using T                          = float;
    using T_FLAG                     = uint32_t;
#ifndef USING_AVX512
    static constexpr int SIMD_WIDTH  = 8;   
#else
    static constexpr int SIMD_WIDTH  = 16;   
#endif
#else
    using T                          = double;
    using T_FLAG                     = uint64_t;
#ifndef USING_AVX512
    static constexpr int SIMD_WIDTH  = 4;   
#else
    static constexpr int SIMD_WIDTH  = 8;   
#endif
#endif
    using Struct_type                = Elasticity_Solver_Data<T,T_FLAG>;
    static constexpr int d           = 3;
    static constexpr int log2_struct = NextLogTwo<sizeof(Struct_type)>::value;
    static constexpr int log2_page   = 7 + log2_struct;
    using SPG_Allocator              = SPGrid_Allocator<Struct_type,d,log2_page>;
    using Mask_Type                  = SPG_Allocator::Array_mask<T>;
    FEMInterface interface(input, output);
    FEMInputs param = interface.param;
    interface.outputs.success = 0;

    // NOTE: SPGrid creates domain as square as possible.
    // And it always grow z dimension first, than y, and last x.
    // So, make sure param.resolution[2] > param.resolution[1] > param.resolution[0] for optimal allocation

    SolverState *state_ptr = static_cast<SolverState *>(interface.param.get_solver_state_ptr());

    bool warm_starting;
    if (state_ptr) {
      // Start from previous solution
      warm_starting = true;
      std::cout<<"Reusing grid: ["<<param.resolution[0]<<", "<<param.resolution[1]<<", "<<param.resolution[2]<<"]"<<std::endl;
      state_ptr->spgrid_ptr = static_cast<SPG_Allocator *>(state_ptr->spgrid_ptr);
    } else {
      // Start from scratch
      warm_starting = false;
      state_ptr = new SolverState;
      std::cout<<"Allocating grid: ["<<param.resolution[0]<<", "<<param.resolution[1]<<", "<<param.resolution[2]<<"]"<<std::endl;
      // TODO: this will never be destroyed, though reused in most cases
      state_ptr->spgrid_ptr = new SPG_Allocator(param.resolution[0],param.resolution[1],param.resolution[2]);
    }
    SolverState &state = *state_ptr;
    SPG_Allocator &allocator = *static_cast<SPG_Allocator *>(state.spgrid_ptr);

    SPGrid_Page_Map<log2_page> page_map(allocator);
    using block_size = FEMInputs::ScalarGrid::block_size;
    T dx = param.dx;
    T Struct_type::* u_fields[3];
    T Struct_type::* f_fields[3];
    T Struct_type::* s_fields[d];
    T Struct_type::* q_fields[d];
    T Struct_type::* z_fields[d];
    T Struct_type::* mu_field;
    T Struct_type::* lambda_field;

    u_fields[0] = &Struct_type::ch0;u_fields[1] = &Struct_type::ch1;u_fields[2] = &Struct_type::ch2;
    f_fields[0] = &Struct_type::ch3;f_fields[1] = &Struct_type::ch4;f_fields[2] = &Struct_type::ch5;
    s_fields[0] = &Struct_type::ch6 ; s_fields[1] = &Struct_type::ch7 ;s_fields[2] = &Struct_type::ch8;
    q_fields[0] = &Struct_type::ch9 ; q_fields[1] = &Struct_type::ch10;q_fields[2] = &Struct_type::ch11;
    z_fields[0] = &Struct_type::ch9; z_fields[1] = &Struct_type::ch10;z_fields[2] = &Struct_type::ch11;
    mu_field = &Struct_type::ch12;lambda_field = &Struct_type::ch13;

    std::cout<<"dx           : "<<param.dx<<std::endl;
    std::cout<<"Global mu    : "<<param.global_mu<<std::endl;
    std::cout<<"Global lambda: "<<param.global_lambda<<std::endl;
    T max_density = 0;
    T min_density = std::numeric_limits<T>::max();
    param.prepare_stiffness_mapping();

    const uint32_t nblocks = param.density.blocks.size();
    std::vector<std::array<int, 3>> base_coordinates;
    if(param.use_density_only){
        auto mu_array = allocator.Get_Array(mu_field);
        auto lambda_array = allocator.Get_Array(lambda_field);
        for(int b = 0;b < nblocks;++b){
            auto& block = param.density.blocks[b];
            std::array<int,d> base_index{block.base_coordinates[0],block.base_coordinates[1],block.base_coordinates[2]};
            base_coordinates.emplace_back(base_index);
        }
        #pragma omp parallel for reduction(max: max_density), reduction(min: min_density)
        for(int b = 0;b < nblocks;++b){
            auto& block = param.density.blocks[b];
            std::array<int,d> base_index{block.base_coordinates[0],block.base_coordinates[1],block.base_coordinates[2]};
            page_map.Set_Page(Mask_Type::Linear_Offset(base_index));
            for(int ii = 0; ii < block_size::x; ii++){
            for(int jj = 0; jj < block_size::y; jj++){
            for(int kk = 0; kk < block_size::z; kk++){
                std::array<int,d> index{block.base_coordinates[0] + ii,
                                        block.base_coordinates[1] + jj,
                                        block.base_coordinates[2] + kk};
                uint64_t offset      = Mask_Type::Linear_Offset(index);
                if(block.get(ii, jj, kk)>0){
                    T d = param.stiffness_mapping(block.get(ii, jj, kk)); // SIMP now happens at the solver side
                    max_density = (d > max_density) ? d : max_density;
                    min_density = (d < min_density) ? d : min_density;
                    mu_array(offset)     = d * param.global_mu;
                    lambda_array(offset) = d * param.global_lambda;}
            }}}}
    }else{
        std::cerr<<"Unsupported mode!"<<std::endl;
        return;
        //TODO load mu and lambda.
    }
    param.density.clear();

    page_map.Update_Block_Offsets();
    std::cout<<"Number of blocks: "<<page_map.Get_Blocks().second<<std::endl;
    {
        Validate_Blocks<Struct_type,log2_page,d> validator;
        if(!validator.Validate_With_Density(allocator,page_map,mu_field)){
            std::cerr<<"(SPGrid_CPU_Solver.cpp) Block activation check has faild...Aborting..."<<std::endl;
            return;}
    }

    // Now, setup the dirichlet condition
    std::vector<unsigned long> dirichlet_offsets[d];
    std::vector<T> dirichlet_values[d];
    for(int i = 0;i < param.dirichlet_nodes.size();++i){
        auto dirichlet = param.dirichlet_nodes[i];
        std::array<int,d> index{dirichlet.coord[0],dirichlet.coord[1],dirichlet.coord[2]};
        uint64_t offset = Mask_Type::Linear_Offset(index);
        dirichlet_offsets[dirichlet.axis].push_back(offset);
        dirichlet_values[dirichlet.axis].push_back(dirichlet.value);}

    CG_VECTOR<Struct_type,T,d,log2_page> u(u_fields, allocator, page_map);
    CG_VECTOR<Struct_type,T,d,log2_page> f(f_fields, allocator, page_map);
    CG_VECTOR<Struct_type,T,d,log2_page> s(s_fields, allocator, page_map);
    CG_VECTOR<Struct_type,T,d,log2_page> q(q_fields, allocator, page_map);
    CG_VECTOR<Struct_type,T,d,log2_page> z(z_fields, allocator, page_map);
    CONJUGATE_GRADIENT_4Ch<T> cg;

    u.using_float = true;
    f.using_float = true;
    s.using_float = true;
    q.using_float = true;
    z.using_float = true;
    // Now convert the Dirichlet_Nodes to Dirichlet_Cells.
    std::vector<unsigned long> dirichlet_cells[d];
    for(int v = 0;v < d;++v){
        std::unordered_set<uint64_t> node_set;
        for(int i = 0;i < dirichlet_offsets[v].size();++i)
            node_set.insert(dirichlet_offsets[v][i]);
        for(int i = 0;i < dirichlet_offsets[v].size();++i){
            const uint64_t neighbor_offsets[] = {
                Mask_Type::Packed_Offset<0,0,1>(dirichlet_offsets[v][i]),
                Mask_Type::Packed_Offset<0,1,0>(dirichlet_offsets[v][i]),
                Mask_Type::Packed_Offset<0,1,1>(dirichlet_offsets[v][i]),
                Mask_Type::Packed_Offset<1,0,0>(dirichlet_offsets[v][i]),
                Mask_Type::Packed_Offset<1,0,1>(dirichlet_offsets[v][i]),
                Mask_Type::Packed_Offset<1,1,0>(dirichlet_offsets[v][i]),
                Mask_Type::Packed_Offset<1,1,1>(dirichlet_offsets[v][i]),
            };
            if(node_set.find(neighbor_offsets[0]) != node_set.end() &&
               node_set.find(neighbor_offsets[1]) != node_set.end() &&
               node_set.find(neighbor_offsets[2]) != node_set.end() &&
               node_set.find(neighbor_offsets[3]) != node_set.end() &&
               node_set.find(neighbor_offsets[4]) != node_set.end() &&
               node_set.find(neighbor_offsets[5]) != node_set.end() &&
               node_set.find(neighbor_offsets[6]) != node_set.end())
                dirichlet_cells[v].push_back(dirichlet_offsets[v][i]);}}

    int force_on_air_counter = 0;
    for(int i = 0;i < param.forces.size();++i){
        ForceOnNode f = param.forces[i];
        std::array<int,d> index{f.coord[0],f.coord[1],f.coord[2]};
        uint64_t offset = Mask_Type::Linear_Offset(index);
        bool has_non_empty_neighbour = false;
        const uint64_t neighbor_offsets[] = {
          Mask_Type::Packed_Offset<-1,-1,-1>(offset),
          Mask_Type::Packed_Offset<-1,-1, 0>(offset),
          Mask_Type::Packed_Offset<-1, 0,-1>(offset),
          Mask_Type::Packed_Offset<-1, 0, 0>(offset),
          Mask_Type::Packed_Offset< 0,-1,-1>(offset),
          Mask_Type::Packed_Offset< 0,-1, 0>(offset),
          Mask_Type::Packed_Offset< 0, 0,-1>(offset),
          Mask_Type::Packed_Offset< 0, 0, 0>(offset),
        };
    }
    std::cout<<"Nodes per block: "<<Mask_Type::elements_per_block<<std::endl;
    auto start_init = std::chrono::steady_clock::now();
    //param.pre_and_post_smoothing_iter=1;
    //param.boundary_smoothing()=3;
    //param.mg_level=5;
    std::cout<<"MG levels              : "<<param.mg_level<<std::endl;
    bool disable_mg = false;
    if (param.mg_level == 0) {
        std::cout<<"MG preconditioner is disabled (mg_level=0). CG will be used"<<std::endl;
        disable_mg = true;
        param.mg_level = 1;
    }
    
    std::cout<<"MG interior smoothing  : "<<param.pre_and_post_smoothing_iter<<std::endl;
    std::cout<<"MG boudnary smoothing  : "<<param.boundary_smoothing()<<std::endl;

    CG_SYSTEM<Struct_type,T,T_FLAG,d,log2_page,SIMD_WIDTH> cg_system(allocator,page_map,mu_field,lambda_field,dx,dirichlet_cells,
                                                                     param.mg_level,//number of mg levels
                                                                     param.pre_and_post_smoothing_iter,//number of smoothing iterations
                                                                     param.boundary_smoothing());
    
    if (disable_mg) {
        delete cg_system.mg;
        cg_system.mg = nullptr;
    }                                                                    

    auto end_init = std::chrono::steady_clock::now();
    auto diff_init = end_init - start_init;
    std::cout<<"Initialization time: "<<double(std::chrono::duration <double, std::milli> (diff_init).count())/1000.0<<"sec"<<std::endl;
    
    bool reuse_u = false;
    for(int v = 0;v < d;++v){
        auto u_array = allocator.Get_Array(u_fields[v]);
        #pragma omp parallel for
        for(int i = 0;i < dirichlet_offsets[v].size();++i){
            u_array(dirichlet_offsets[v][i]) = dirichlet_values[v][i];}}
    if (warm_starting) {
        // Compute residual using previous u*
        // reuse it only if the residual is smaller than that of zero initial guess
        s.Clear();
        #pragma omp parallel for
        for(int i = 0;i < param.forces.size();++i){
            ForceOnNode f = param.forces[i];
            std::array<int,d> index{f.coord[0],f.coord[1],f.coord[2]};
            uint64_t offset = Mask_Type::Linear_Offset(index);
            for(int v = 0;v < d;++v)
                allocator.Get_Array(s_fields[v])(offset) = f.f[v];}

        cg_system.Project(s);
        cg_system.Project_Nullspace(s);
        cg_system.Residual(u,s,f,mu_field,lambda_field);
        cg_system.Project(f);

        q.Clear();
        s.Clear();
        const T norm_linf=cg_system.Convergence_Norm(f);
        printf("warm_starting initial guess (L_inf): %f\n", norm_linf);
        printf("         zero initial guess (L_inf): %f\n", state.initial_residual_linf);
        if (param.forced_reuse() || state.initial_residual_linf > norm_linf) {
            reuse_u = true;
            printf("Reusing u\n");
        } else {
            printf("Discarding u\n");
        }
    }

    if (!reuse_u) {
        u.Clear();
    } // reuse u otherwise

    for(int v = 0;v < d;++v){
        auto u_array = allocator.Get_Array(u_fields[v]);
        #pragma omp parallel for
        for(int i = 0;i < dirichlet_offsets[v].size();++i){
            u_array(dirichlet_offsets[v][i]) = dirichlet_values[v][i];}}

    T initial_residual_linf = 0;
    // For the last iteration, print the residual only
    s.Clear();
    #pragma omp parallel for
    for(int i = 0;i < param.forces.size();++i){
        ForceOnNode f = param.forces[i];
        std::array<int,d> index{f.coord[0],f.coord[1],f.coord[2]};
        uint64_t offset = Mask_Type::Linear_Offset(index);
        for(int v = 0;v < d;++v)
            allocator.Get_Array(s_fields[v])(offset) = f.f[v];}
    
    cg_system.Project(s);
    cg_system.Project_Nullspace(s);
    cg_system.Residual(u,s,f,mu_field,lambda_field);
    cg_system.Project(f);
    
    // Now backup u
    T* u_bk[d];
    for(int v = 0;v < d;++v) u_bk[v] = new T[page_map.Get_Blocks().second*uint64_t(Mask_Type::elements_per_block)];
    for(int v = 0;v < d;++v){
        auto data = allocator.Get_Const_Array(u_fields[v]);
        auto blocks = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            uint64_t array_offset = b * uint64_t(Mask_Type::elements_per_block);
            uint64_t data_offset  = blocks.first[b];
            for(int i = 0; i < Mask_Type::elements_per_block; i++,array_offset++,data_offset+=sizeof(T)){
                u_bk[v][array_offset] = data(data_offset);}}}
    
    q.Clear();
    s.Clear();
    cg.print_residuals=param.krylov.print_residuals;
    cg.print_diagnostics=true;
    const T norm_linf=cg_system.Convergence_Norm(f);
    if (!warm_starting) state.initial_residual_linf = norm_linf;
    std::cout<<"initial residual L_inf  : "<<state.initial_residual_linf<<std::endl;
    {auto start = std::chrono::steady_clock::now();
     u.Clear();
     const T inner_tolorence = T(param.krylov.tolerance*state.initial_residual_linf);
     cg.Solve(cg_system,u,f,s,q,z,inner_tolorence,0,500/*param.krylov.max_iterations*/);
     auto end = std::chrono::steady_clock::now();
     auto diff = end - start;
     std::cout<<"Solve time: "<<double(std::chrono::duration <double, std::milli> (diff).count())/1000.0<<"sec"<<std::endl;
    }

    for(int v = 0;v < d;++v){
        auto data = allocator.Get_Array(u_fields[v]);
        auto blocks = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            uint64_t array_offset = b * uint64_t(Mask_Type::elements_per_block);
            uint64_t data_offset  = blocks.first[b];
            for(int i = 0; i < Mask_Type::elements_per_block; i++,array_offset++,data_offset+=sizeof(T)){
                data(data_offset) += u_bk[v][array_offset];}}}
    for(int v = 0;v < d;++v) delete [] u_bk[v];
    
    if(param.use_density_only){        
        interface.outputs.displacements.blocks.resize(nblocks);
        for(int b = 0;b < nblocks;++b){
            FEMOutputs::VectorGrid::Block displacement_block;
            for(int v = 0; v < d; v++){
                displacement_block.base_coordinates[v] = base_coordinates[b][v];}
            for(int ii = 0;ii < block_size::x;ii++){
            for(int jj = 0;jj < block_size::y;jj++){
            for(int kk = 0;kk < block_size::z;kk++){
                std::array<int,d> index{base_coordinates[b][0] + ii,
                        base_coordinates[b][1] + jj,
                        base_coordinates[b][2] + kk};
                uint64_t offset = Mask_Type::Linear_Offset(index);
                for(int v = 0; v < d; v++){
                    displacement_block.get(ii,jj,kk)[v] = allocator.Get_Array(u_fields[v])(offset);}
                T residual_norm = 0;
                for(int v = 0; v < d; v++){
                    residual_norm = std::max(residual_norm, std::abs(allocator.Get_Array(f_fields[v])(offset)));}
                displacement_block.get(ii,jj,kk)[3] = residual_norm;
              }}}
            interface.outputs.displacements.blocks[b] = displacement_block;
        }
    }else{
        //TODO
    }
    interface.outputs.success = 1;

    s.Clear();
    #pragma omp parallel for
    for(int i = 0;i < param.forces.size();++i){
        ForceOnNode f = param.forces[i];
        std::array<int,d> index{f.coord[0],f.coord[1],f.coord[2]};
        uint64_t offset = Mask_Type::Linear_Offset(index);
        for(int v = 0;v < d;++v)
            allocator.Get_Array(s_fields[v])(offset) = f.f[v];}

    cg_system.Project(s);
    cg_system.Project_Nullspace(s);
    cg_system.Residual(u,s,f,mu_field,lambda_field);
    cg_system.Project(f);

    q.Clear();
    s.Clear();
    T norm_linf_final=cg_system.Convergence_Norm(f);
    printf("Final residual (recomputed, L_inf): %G\n", norm_linf_final);
    
    // Save the allocator
    if (param.keep_state()) {
        // Clear density so that it will not fail the activation test
        printf("Clearing density...\n");
        auto blocks = page_map.Get_Blocks();
        auto mu_array = allocator.Get_Array(mu_field);
        auto lambda_array = allocator.Get_Array(lambda_field);
        auto flags_array = allocator.Get_Array(&Struct_type::flags);
        #pragma omp parallel for
        for (uint64_t i = 0; i < blocks.second; i++) {
          auto offset = blocks.first[i];
          for(int e=0;e<Mask_Type::elements_per_block;e++,offset+=sizeof(T)){
              mu_array(offset) = 0;
              lambda_array(offset) = 0;
              flags_array(offset) = 0;}}
        interface.outputs.set_solver_state_ptr(&state);
    } else {
        delete static_cast<SPG_Allocator *>(state.spgrid_ptr);
        delete state_ptr;
    }
}
