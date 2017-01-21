@[Link("libsvm")]

lib LibSVM
  # svm code
  struct Svm_node
    index : Int32
    value : LibC::Double
  end

  struct Svm_problem
    l : Int32
    y : LibC::Double*
    x : Svm_node**
  end

  struct Svm_parameter
    svm_type : Int32
    kernel_type : Int32
    degree : Int32       # for poly
    gamma : LibC::Double # for poly/rbf/sigmoid
    coef0 : LibC::Double # for poly/sigmoid

    # these are for training only #
    cache_size : LibC::Double # in MB
    eps : LibC::Double        # stopping criteria
    c : LibC::Double          # for C_SVC, EPSILON_SVR, and NU_SVR
    nr_weight : Int32         # for C_SVC
    weight_label : Int32*     # for C_SVC
    weight : LibC::Double*    # for C_SVC
    nu : LibC::Double         # for NU_SVC, ONE_CLASS, and NU_SVR
    p : LibC::Double          # for EPSILON_SVR
    shrinking : Int32         # use the shrinking heuristics
    probability : Int32       # do probability estimates
  end

  struct Svm_model
    param : Svm_parameter    # parameter
    nr_class : Int32         # number of classes, = 2 in regression/one class svm
    l : Int32                # total #SV
    sv : Svm_node**          # SVs (SV[l])
    sv_coef : LibC::Double** # coefficients for SVs in decision functions (sv_coef[k-1][l])
    rho : LibC::Double*      # constants in decision functions (rho[k*(k-1)/2])
    probA : LibC::Double*    # pairwise probability information
    probB : LibC::Double*
    sv_indices : Int32* # sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

    # for classification only #

    label : Int32* # label of each class (label[k])
    nSV : Int32*   # number of SVs for each class (nSV[k])
    # nSV[0] + nSV[1] + ... + nSV[k-1] = l

    free_sv : Int32 # 1 if svm_model is created by svm_load_model
    # 0 if svm_model is created by svm_train
  end

  fun svm_train(prob : Svm_problem*, param : Svm_parameter*) : Svm_model*
  fun svm_predict(model : Svm_model*, x : Svm_node*) : LibC::Double
  fun svm_cross_validation(prob : Svm_problem*, param : Svm_parameter*, nr_fold : Int32, target : LibC::Double*) : Void
  fun svm_get_svm_type(model : Svm_model*) : Int32
  fun svm_get_nr_class(model : Svm_model*) : Int32
  fun svm_get_labels(model : Svm_model*, label : Int32*) : Void
  fun svm_get_sv_indices(model : Svm_model*, sv_indices : Int32*) : Void
  fun svm_get_nr_sv(model : Svm_model*) : Int32
  fun svm_get_svr_probability(model : Svm_model*) : LibC::Double
  fun svm_predict_values(model : Svm_model*, x : Svm_node*, dec_values : LibC::Double*) : LibC::Double
  fun svm_predict_probability(model : Svm_model*, x : Svm_node*, prob_estimates : LibC::Double*) : LibC::Double
  fun svm_check_parameter(prob : Svm_problem*, param : Svm_parameter*) : String
  fun svm_check_probability_model(model : Svm_model*) : Int32
  fun svm_save_model(model_file_name : String, model : Svm_model*) : Int32
  fun svm_load_model(model_file_name : String) : Svm_model*
  fun svm_free_model_content(model_ptr : Svm_model*) : Void
  fun svm_free_and_destroy_model(model_ptr_ptr : Svm_model**) : Void
  fun svm_destroy_param(param : Svm_parameter*) : Void
  fun svm_set_print_string_function(print_func : Pointer(T) | String) : Void
end
