import unittest

class TestModel(unittest.TestCase):
    
    def __init__(self, test_name, extra_params):
        super(TestModel, self).__init__(test_name)  # calling the super class init varies for different python versions.  This works for 2.7
        self.extra_params = extra_params
    
    def test_shape_input(self):
        x_data_shape = self.extra_params["x_data"].shape[1:]
        excepted_shape = (self.extra_params["window"],self.extra_params["num_features"])
        self.assertEqual(x_data_shape, excepted_shape, f"Should be ({self.extra_params['window']}, {self.extra_params['num_features']})")

    def test_shape_output(self):
        y_data_shape = self.extra_params["y_data"].shape[1:]
        excepted_shape = (1,)
        self.assertEqual(y_data_shape, excepted_shape, "Should be (1,)")
        
    def test_model_input(self):
        model_input_shape = tuple(self.extra_params["model"].layers[0].input.get_shape().as_list())[1:]
        excepted_shape = (self.extra_params["window"],self.extra_params["num_features"])
        self.assertEqual(model_input_shape, excepted_shape, f"Should be ({self.extra_params['window']}, {self.extra_params['num_features']})")
        
    def test_model_output(self):
        model_output_shape = tuple(self.extra_params["model"].layers[-1].output.get_shape().as_list())[1:]
        excepted_shape = (self.extra_params["num_classes"],)
        self.assertEqual(model_output_shape, excepted_shape, "Should be (3,)")
        
    def test_prediction_shape(self):
        num_samples = 3
        x_test_sample = self.extra_params["x_data"][0:num_samples,:,:]
        y_pred_sample = self.extra_params["model"].predict(x_test_sample)
        model_pred_shape = y_pred_sample.shape
        excepted_shape = (num_samples, self.extra_params["num_classes"])
        self.assertEqual(model_pred_shape, excepted_shape, f"Should be ({num_samples}, {self.extra_params['num_classes']}")

def execute_test(test_names, mlflow, extra_params):
    test_result = []
    
    for test in test_names:
        try:
            suite = unittest.TestSuite()
            suite.addTest(TestModel(test, extra_params))
            runner = unittest.TextTestRunner()
            runner.run(suite)
            mlflow.log_param(test, "Passed")
            test_result.append(True)
        except Exception as e:
            mlflow.log_param(test, "Failed")
            test_result.append(False)
            print(e)
    return test_result