public class OutlookAddIn
{
    private void ThisAddIn_Startup(object sender, System.EventArgs e)
    {
        // Handle new email events
        Application.NewMailEx += new ApplicationEvents_11_NewMailExEventHandler(HandleNewMail);
    }

    private void HandleNewMail(string EntryIDCollection)
    {
        // Get the new mail item
        Outlook.MailItem mailItem = Application.Session.GetItemFromID(EntryIDCollection);
        // Process mail item using your classification model
        // Move to appropriate folder based on classification
        mailItem.Move(targetFolder);
    }
}
