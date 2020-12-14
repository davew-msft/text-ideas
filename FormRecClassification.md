

When using the Form Recognizer service, there is typically a 1:1 correlation between the forms and their respective layout and the model that they were trained on. In other words, for many forms with the same layout, a single model will exist that can be used for that layout.

Thus in order to be able to run evaluation against a form to extract the fields required, the form needs to be routed to the correct model. This identification of the form and routing it to the correct model is referred to here as the classification step, as we need to classify the form layout to route it to the appropriate model associated with that layout.

This is clearly a crucial step in the process as if this step performs poorly, all other steps will too.