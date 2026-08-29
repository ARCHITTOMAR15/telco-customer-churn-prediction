import sys

class CustomException(Exception):
    """
    Custom exception class for the Telco Customer Churn project.
    Provides detailed error messages including filename and line number.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):

        _, _, exc_tb = error_detail.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        detailed_error = (
            f"\nError occurred in Python script : [{file_name}]"
            f"\nLine Number                 : [{line_number}]"
            f"\nError Message               : [{error_message}]"
        )

        return detailed_error

    def __str__(self):
        return self.error_message