import React from "react";
import styled from "styled-components";

const StyledSection03 = styled.div`
  width: 1100px;
  height: 800px;
  max-width: 1000px;
  padding: 50px 48px 120px 48px;
  display: inline-flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 43px;
`;

const StyledFrame1 = styled.div`
  width: 435px;
  display: inline-flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 20px;
`;

const StyledFrame101 = styled.div`
  align-self: stretch;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 5px;
`;

const StyledSubtitle = styled.div`
  align-self: stretch;
  color: black;
  font-size: 11px;
  font-family: "Poppins", sans-serif;
  font-weight: 300;
  line-height: 15.4px;
  letter-spacing: 1.1px;
  word-wrap: break-word;
`;

const StyledTitle = styled.div`
  align-self: stretch;
  color: black;
  font-size: 90px;
  font-family: "Spectral", serif;
  font-weight: 300;
  line-height: 97.2px;
  letter-spacing: 1.17px;
  word-wrap: break-word;
`;

const StyledParagraph = styled.div`
  align-self: stretch;
  color: black;
  font-size: 17px;
  font-family: "Brygada 1918", serif;
  font-style: italic;
  font-weight: 400;
  line-height: 30px;
  word-wrap: break-word;
`;

const StyledImage = styled.img`
  width: 521px;
  height: 688px;
`;

export const Section03 = ({ subtitle, title, body, imageUrl }) => {
  return (
    <StyledSection03>
      <StyledFrame1>
        <StyledFrame101>
          <StyledSubtitle>{subtitle}</StyledSubtitle>

          <StyledTitle>
            {title.split("\n").map((line, i) => (
              <React.Fragment key={i}>
                {line}
                <br />
              </React.Fragment>
            ))}
          </StyledTitle>
        </StyledFrame101>

        <StyledParagraph>
          {body.split("\n").map((line, i) => (
            <React.Fragment key={i}>
              {line}
              <br />
            </React.Fragment>
          ))}
        </StyledParagraph>
      </StyledFrame1>

      <StyledImage src={imageUrl} alt="Section image" />
    </StyledSection03>
  );
};
