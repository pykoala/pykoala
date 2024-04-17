"""
Main exceptions used to trace errors within pykoala.
"""


class NoneAttrError(Exception):
    """
    Exception raised when trying to access to an attributed not provided.
    """
    def __init__(self, attr_name):
        self.message = ("Attribute {} not provided (None)".format(attr_name))
        super().__init__(self.message)


class ClassError(Exception):
    """
        Exception raised when providing wrong data
    """

    def __init__(self, expected_classes, input_value):
        expected_classes_string = ' {}\n' * len(expected_classes)
        self.message = ("Input object class {} ".format(input_value) +
                        "does not match expected classes:\n" + expected_classes_string.format(
                        *expected_classes))
        super().__init__(self.message)


class MaskError(Exception):
    """
    Exceptions raised for errors during data masking.
    """


class MaskBitError(MaskError):
    """
    Exception raised when providing wrong bit mask value
    """
    def __init__(self, accepted_values, input_value):
        self.message = "Mask value {} does not correspond default: {}"\
            .format(accepted_values, ', '.join(str(i) for i in input_value))
        super().__init__(self.message)


class CorrectionClassError(Exception):
    """
    Exception class raised for CorrectionBase classes
    """
    def __init__(self, target_class, wrong_target):
        self.message = "ERROR: Target class {} does not match input {}".format(target_class, wrong_target)
        super().__init__(self.message)


class TelluricError(Exception):
    """
    Parent exception class raised during telluric correction
    """


class TelluricNoFileError(TelluricError):
    """
    Parent exception class raised during telluric correction
    """
    def __init__(self):
        self.message = "ERROR: Not RSS nor Telluric file provided"
        super().__init__(self.message)


class FitError(Exception):
    """
    Parent exception class raised during telluric correction
    """
    def __init__(self):
        self.message = "Unsuccessful fit."
        super().__init__(self.message)


class CalibrationError(Exception):
    """
        Exception class raised during flux calibration
        """

    def __init__(self):
        self.message = "Data Container already flux calibrated."
        super().__init__(self.message)
